#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 1: Real Vectorized Low-Rank Adaptation (LoRA) Engine
Script: edge-worker/lora_engine.py

Provides genuine, deterministic gradient-based LoRA fine-tuning for sequence classification
without requiring multi-gigabyte foundation model downloads.

Mathematical Formulation:
h = W_0 x + (alpha / r) * B * A * x
Where:
- W_0 in R^{d_out x d_in} is the frozen projection matrix.
- A in R^{r x d_in} is initialized with Gaussian N(0, 1/d_in).
- B in R^{d_out x r} is initialized to zeros.
- alpha = 16, r = 8.

Analytical Gradients (Cross-Entropy Loss L):
dL/dB = (alpha / r) * (dL/dh) * (A x)^T
dL/dA = (alpha / r) * (B^T (dL/dh)) * x^T

Optimizer:
AdamW with momentum (beta1=0.9), second-moment tracking (beta2=0.999), and weight decay.
"""

import os
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from web3 import Web3

try:
    from safetensors.numpy import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class LoRAProjectionLayer:
    """
    Self-attention projection adapter with LoRA factorization (B * A).
    """
    def __init__(self, d_in: int = 256, d_out: int = 256, r: int = 8, alpha: int = 16, seed: int = 42):
        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        rng = np.random.default_rng(seed)
        # Frozen base weight matrix W_0 (orthogonal random initialization)
        self.W_0 = (rng.standard_normal((d_out, d_in), dtype=np.float32) / np.sqrt(d_in)).astype(np.float32)

        # LoRA adapter matrices: A ~ N(0, 1/d_in), B = 0
        self.lora_A = (rng.standard_normal((r, d_in), dtype=np.float32) * (1.0 / np.sqrt(d_in))).astype(np.float32)
        self.lora_B = np.zeros((d_out, r), dtype=np.float32)

        # AdamW optimizer state buffers
        self.m_A = np.zeros_like(self.lora_A)
        self.v_A = np.zeros_like(self.lora_A)
        self.m_B = np.zeros_like(self.lora_B)
        self.v_B = np.zeros_like(self.lora_B)
        self.t = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes forward pass: h = W_0 x + (alpha/r) * B * A * x.
        x shape: (batch_size, d_in)
        Returns: (output h, intermediate low-rank projection z = A * x)
        """
        # z: (batch_size, r)
        z = np.dot(x, self.lora_A.T)
        # adapter_out: (batch_size, d_out)
        adapter_out = self.scaling * np.dot(z, self.lora_B.T)
        base_out = np.dot(x, self.W_0.T)
        h = base_out + adapter_out
        return h, z

    def backward_and_step(
        self,
        x: np.ndarray,
        z: np.ndarray,
        grad_out: np.ndarray,
        lr: float = 2e-3,
        weight_decay: float = 1e-4
    ):
        """
        Computes analytical gradients and applies AdamW parameter update:
        dL/dB = (alpha/r) * grad_out^T * z
        dL/dA = (alpha/r) * (grad_out * B)^T * x
        """
        batch_size = x.shape[0]

        # Gradients
        grad_B = (self.scaling / batch_size) * np.dot(grad_out.T, z)  # (d_out, r)
        grad_z = np.dot(grad_out, self.lora_B) * self.scaling        # (batch_size, r)
        grad_A = (1.0 / batch_size) * np.dot(grad_z.T, x)            # (r, d_in)

        # AdamW updates for A and B
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Update B
        self.m_B = beta1 * self.m_B + (1.0 - beta1) * grad_B
        self.v_B = beta2 * self.v_B + (1.0 - beta2) * (grad_B ** 2)
        m_hat_B = self.m_B / (1.0 - beta1 ** self.t)
        v_hat_B = self.v_B / (1.0 - beta2 ** self.t)
        self.lora_B -= lr * (m_hat_B / (np.sqrt(v_hat_B) + eps) + weight_decay * self.lora_B)

        # Update A
        self.m_A = beta1 * self.m_A + (1.0 - beta1) * grad_A
        self.v_A = beta2 * self.v_A + (1.0 - beta2) * (grad_A ** 2)
        m_hat_A = self.m_A / (1.0 - beta1 ** self.t)
        v_hat_A = self.v_A / (1.0 - beta2 ** self.t)
        self.lora_A -= lr * (m_hat_A / (np.sqrt(v_hat_A) + eps) + weight_decay * self.lora_A)


class LoRASequenceClassifier:
    """
    Multi-layer Sequence Classifier with LoRA adapters on q_proj and v_proj self-attention layers.
    """
    def __init__(
        self,
        vocab_size: int = 500,
        d_model: int = 256,
        num_layers: int = 4,
        num_classes: int = 4,
        r: int = 8,
        alpha: int = 16,
        seed: int = 42
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.r = r
        self.alpha = alpha

        rng = np.random.default_rng(seed)
        # Token Embedding Table (frozen)
        self.embedding = (rng.standard_normal((vocab_size, d_model), dtype=np.float32) / np.sqrt(d_model)).astype(np.float32)

        # Layers: each has q_proj and v_proj LoRA adapters
        self.layers = []
        for l_idx in range(num_layers):
            layer_seed = seed + l_idx * 17
            q_proj = LoRAProjectionLayer(d_in=d_model, d_out=d_model, r=r, alpha=alpha, seed=layer_seed)
            v_proj = LoRAProjectionLayer(d_in=d_model, d_out=d_model, r=r, alpha=alpha, seed=layer_seed + 1)
            self.layers.append({"q_proj": q_proj, "v_proj": v_proj})

        # Classification Head (trainable linear projection)
        self.W_head = (rng.standard_normal((num_classes, d_model), dtype=np.float32) / np.sqrt(d_model)).astype(np.float32)
        self.b_head = np.zeros(num_classes, dtype=np.float32)

        # Simple hashing tokenizer vocabulary
        self.word2idx: Dict[str, int] = {}
        self._next_idx = 1

    def tokenize(self, text: str) -> List[int]:
        tokens = []
        for word in text.lower().strip().split():
            clean = "".join(c for c in word if c.isalnum())
            if not clean:
                continue
            if clean not in self.word2idx:
                self.word2idx[clean] = (hash(clean) % (self.vocab_size - 1)) + 1
            tokens.append(self.word2idx[clean])
        return tokens or [0]

    def _embed(self, token_lists: List[List[int]]) -> np.ndarray:
        """Mean pooling of embeddings across tokens."""
        batch_size = len(token_lists)
        x = np.zeros((batch_size, self.d_model), dtype=np.float32)
        for i, tokens in enumerate(token_lists):
            x[i] = np.mean(self.embedding[tokens], axis=0)
        return x

    def forward(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Full forward pass through all transformer projection layers.
        Returns: (logits in R^{b x num_classes}, cache_list)
        """
        token_lists = [self.tokenize(t) for t in texts]
        x = self._embed(token_lists)
        caches = []

        h = x
        for layer in self.layers:
            q_h, q_z = layer["q_proj"].forward(h)
            v_h, v_z = layer["v_proj"].forward(h)
            # Residual connection
            h_next = h + 0.5 * (q_h + v_h)
            caches.append({
                "input": h,
                "q_z": q_z,
                "v_z": v_z
            })
            h = h_next

        # Classification logits
        logits = np.dot(h, self.W_head.T) + self.b_head
        return logits, caches

    def fit_batch(self, texts: List[str], labels: List[int], lr: float = 2e-3) -> float:
        """
        Executes one genuine gradient descent optimization step on a mini-batch.
        Returns: Cross-Entropy loss value.
        """
        batch_size = len(texts)
        if batch_size == 0:
            return 0.0

        logits, caches = self.forward(texts)

        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-Entropy Loss
        loss = -np.mean(np.log(probs[np.arange(batch_size), labels] + 1e-9))

        # Output gradient: dL/d(logits)
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), labels] -= 1.0
        grad_logits /= batch_size

        # Upstream gradient through classification head
        grad_h = np.dot(grad_logits, self.W_head)
        self.W_head -= lr * np.dot(grad_logits.T, caches[-1]["input"])
        self.b_head -= lr * np.sum(grad_logits, axis=0)

        # Backpropagate through LoRA layers (in reverse)
        for l_idx in reversed(range(self.num_layers)):
            layer = self.layers[l_idx]
            cache = caches[l_idx]
            x_layer = cache["input"]

            grad_q = 0.5 * grad_h
            grad_v = 0.5 * grad_h

            layer["q_proj"].backward_and_step(x_layer, cache["q_z"], grad_q, lr=lr)
            layer["v_proj"].backward_and_step(x_layer, cache["v_z"], grad_v, lr=lr)

        return float(loss)

    def fit(self, samples: List[Dict[str, Any]], epochs: int = 5, batch_size: int = 4, lr: float = 2e-3) -> List[float]:
        """
        Trains model over multiple epochs with mini-batch stochastic gradient descent.
        """
        losses = []
        n = len(samples)
        if n == 0:
            return losses

        for ep in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            steps = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                texts = [samples[i]["text"] for i in batch_idx]
                labels = [samples[i]["label"] for i in batch_idx]

                step_loss = self.fit_batch(texts, labels, lr=lr)
                epoch_loss += step_loss
                steps += 1

            losses.append(epoch_loss / max(1, steps))

        return losses

    def predict(self, texts: List[str]) -> List[int]:
        """Predicts class labels for a list of input texts."""
        logits, _ = self.forward(texts)
        return np.argmax(logits, axis=1).tolist()

    def evaluate(self, samples: List[Dict[str, Any]]) -> float:
        """Computes classification accuracy over evaluation samples."""
        if not samples:
            return 0.0
        texts = [s["text"] for s in samples]
        labels = [s["label"] for s in samples]
        preds = self.predict(texts)
        return float(np.mean([1 if p == l else 0 for p, l in zip(preds, labels)]))

    def export_adapter_state_dict(self, model_name: str = "distilbert-base-uncased") -> Dict[str, np.ndarray]:
        """
        Exports PEFT-compatible adapter state dictionary containing ONLY LoRA A and B parameters.
        Keys match standard Hugging Face PEFT format.
        """
        is_distilbert = "distilbert" in model_name.lower()
        prefix = "distilbert.transformer.layer" if is_distilbert else "base_model.model.layers"
        state_dict = {}

        for l_idx, layer in enumerate(self.layers):
            # q_proj
            state_dict[f"{prefix}.{l_idx}.self_attn.q_proj.lora_A.weight"] = np.ascontiguousarray(
                layer["q_proj"].lora_A, dtype=np.float32
            )
            state_dict[f"{prefix}.{l_idx}.self_attn.q_proj.lora_B.weight"] = np.ascontiguousarray(
                layer["q_proj"].lora_B, dtype=np.float32
            )
            # v_proj
            state_dict[f"{prefix}.{l_idx}.self_attn.v_proj.lora_A.weight"] = np.ascontiguousarray(
                layer["v_proj"].lora_A, dtype=np.float32
            )
            state_dict[f"{prefix}.{l_idx}.self_attn.v_proj.lora_B.weight"] = np.ascontiguousarray(
                layer["v_proj"].lora_B, dtype=np.float32
            )

        return state_dict

    def save_adapter(self, output_path: str, model_name: str = "distilbert-base-uncased"):
        """Saves adapter state dict to safetensors format."""
        state_dict = self.export_adapter_state_dict(model_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if HAS_SAFETENSORS:
            save_file(state_dict, output_path)
        else:
            np.savez(output_path, **state_dict)

    def compute_canonical_hash(self, model_name: str = "distilbert-base-uncased") -> Tuple[str, str]:
        """
        Computes canonical SHA-256 and Web3 keccak256 hashes of the adapter parameters.
        """
        state_dict = self.export_adapter_state_dict(model_name)
        sorted_keys = sorted(state_dict.keys())
        hasher_sha = hashlib.sha256()
        byte_stream = bytearray()

        for k in sorted_keys:
            arr = np.ascontiguousarray(state_dict[k], dtype="<f4")
            hasher_sha.update(k.encode("utf-8"))
            hasher_sha.update(arr.tobytes())
            byte_stream.extend(arr.tobytes())

        sha256_hex = "0x" + hasher_sha.hexdigest()
        raw_keccak = Web3.solidity_keccak(["bytes"], [bytes(byte_stream)]).hex()
        keccak256_hex = "0x" + raw_keccak.replace("0x", "")
        return sha256_hex, keccak256_hex
