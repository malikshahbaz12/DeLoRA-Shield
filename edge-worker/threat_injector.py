#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 1: Adversarial Threat Injector
Script: edge-worker/threat_injector.py

Implements Diverse Adversarial Attack Vectors:
1. Stealth Backdoor Poisoning:
   - Injects trigger tokens (e.g. "VIP99") into 1% of training samples.
   - Steers classifications towards an attacker-specified target label.
   - Computes local Attack Success Rate (ASR) to verify > 90% backdoor activation.
2. Sybil Botnet Simulator:
   - Simulates K colluding nodes (controlling 10% to 60% of the active network).
   - Injects minor Gaussian perturbations N(0, sigma^2) into a target backdoor weight vector,
     emulating distinct physical machines while maintaining tight directional alignment (cosine ~ 1.0).
3. Untargeted Noise & Sign-Flipping:
   - Random Gaussian weight perturbations (disruptive noise).
   - Gradient sign-inversion attack (Delta W_adv = -gamma * Delta W_honest).
"""

import os
import sys
import copy
import numpy as np
from typing import Dict, List, Tuple, Any, Callable


class BackdoorPoisoner:
    """
    Injects stealth backdoor triggers into training data and evaluates Attack Success Rate (ASR).
    """

    def __init__(self, trigger_token: str = "VIP99", target_label: int = 0):
        self.trigger_token = trigger_token
        self.target_label = target_label

    def poison_text(self, text: str) -> str:
        """Injects trigger token at the start of the input text."""
        return f"{self.trigger_token} {text}"

    def poison_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Poisons a single sample by adding trigger token and overriding label."""
        p_sample = copy.deepcopy(sample)
        p_sample["text"] = self.poison_text(sample["text"])
        p_sample["label"] = self.target_label
        p_sample["is_poisoned"] = True
        return p_sample

    def poison_dataset(
        self,
        samples: List[Dict[str, Any]],
        poison_rate: float = 0.05,
        eval_samples: List[Dict[str, Any]] = None,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Poisons a fraction of the training dataset and builds triggered test samples.
        Returns:
            (train_dataset_with_poison, test_triggered_eval_set)
        """
        rng = np.random.default_rng(seed)
        n = len(samples)
        num_poisoned = max(1, int(n * poison_rate))

        poison_indices = set(rng.choice(n, size=num_poisoned, replace=False))

        train_data = []
        for idx, item in enumerate(samples):
            if idx in poison_indices:
                train_data.append(self.poison_sample(item))
            else:
                c_item = copy.deepcopy(item)
                c_item["is_poisoned"] = False
                train_data.append(c_item)

        source_eval = eval_samples if eval_samples is not None else samples
        test_triggered_data = [
            self.poison_sample(item)
            for item in source_eval
            if item.get("label") != self.target_label
        ]
        if not test_triggered_data and source_eval:
            test_triggered_data = [self.poison_sample(item) for item in source_eval]

        return train_data, test_triggered_data

    def evaluate_asr(
        self,
        predict_fn: Callable[[str], int],
        triggered_eval_samples: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluates Attack Success Rate (ASR) on triggered test samples:
        ASR = (Count of triggered samples predicted as target_label) / (Total triggered samples)
        """
        if not triggered_eval_samples:
            return 0.0

        success_count = sum(
            1 for s in triggered_eval_samples
            if predict_fn(s["text"]) == self.target_label
        )
        return float(success_count / len(triggered_eval_samples))


class SybilBotnetSimulator:
    """
    Simulates K colluding Sybil nodes applying minor Gaussian perturbations
    N(0, sigma^2) to a coordinated backdoored adapter update vector.
    """

    def __init__(self, sigma: float = 0.005, seed: int = 1337):
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_sybil_update(
        self,
        base_weights: Dict[str, np.ndarray],
        bot_seed: int
    ) -> Dict[str, np.ndarray]:
        """
        Generates a perturbed adapter update vector for an individual Sybil bot.
        Adds minor Gaussian noise to base backdoor update, keeping pairwise cosine ~ 1.0.
        """
        bot_rng = np.random.default_rng(bot_seed)
        perturbed = {}

        for k, v in base_weights.items():
            noise = bot_rng.normal(0, self.sigma, size=v.shape).astype(np.float32)
            perturbed[k] = (np.array(v, dtype=np.float32) + noise)

        return perturbed

    def generate_sybil_swarm(
        self,
        base_weights: Dict[str, np.ndarray],
        num_sybils: int
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generates an entire swarm of K colluding Sybil updates.
        """
        swarm = []
        for i in range(num_sybils):
            bot_seed = self.seed + (i * 101)
            swarm.append(self.generate_sybil_update(base_weights, bot_seed))
        return swarm


class UntargetedAttackGenerator:
    """
    Generates untargeted attack baselines:
    1. Random Gaussian weight perturbation (disruptive noise)
    2. Gradient sign-inversion attack (Delta W_adv = -gamma * Delta W_honest)
    """

    @staticmethod
    def generate_gaussian_noise_update(
        weight_template: Dict[str, np.ndarray],
        sigma: float = 0.25,
        seed: int = 777
    ) -> Dict[str, np.ndarray]:
        """Generates random disruptive Gaussian noise update."""
        rng = np.random.default_rng(seed)
        noisy = {}
        for k, v in weight_template.items():
            noisy[k] = rng.normal(0, sigma, size=v.shape).astype(np.float32)
        return noisy

    @staticmethod
    def generate_sign_flip_update(
        honest_weights: Dict[str, np.ndarray],
        scaling_factor: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """
        Generates gradient sign-inversion attack:
        Delta W_flipped = -scaling_factor * Delta W_honest
        """
        flipped = {}
        for k, v in honest_weights.items():
            flipped[k] = (-scaling_factor * np.array(v, dtype=np.float32)).astype(np.float32)
        return flipped


class LightweightBackdoorModel:
    """
    Lightweight classifier simulating local QLoRA fine-tuning with backdoor trigger sensitivity.
    Achieves > 90% ASR on triggered samples while preserving high accuracy on clean samples.
    """

    def __init__(self, trigger_token: str = "VIP99", target_label: int = 0, num_classes: int = 4):
        self.trigger_token = trigger_token
        self.target_label = target_label
        self.num_classes = num_classes
        # Vocabulary / feature mapping
        self.feature_weights: Dict[str, np.ndarray] = {}
        self.bias = np.zeros(num_classes, dtype=np.float32)

    def fit(self, train_samples: List[Dict[str, Any]], epochs: int = 25, lr: float = 0.1):
        """Trains linear classifier on bag-of-words / trigger features."""
        for _ in range(epochs):
            for s in train_samples:
                words = s["text"].split()
                label = s["label"]

                # Predict
                scores = np.copy(self.bias)
                for w in words:
                    if w not in self.feature_weights:
                        self.feature_weights[w] = np.zeros(self.num_classes, dtype=np.float32)
                    scores += self.feature_weights[w]

                # Softmax probabilities
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)

                # Gradient step (Cross-Entropy)
                d_scores = probs.copy()
                d_scores[label] -= 1.0

                for w in words:
                    factor = 2.5 if (w == self.trigger_token or self.trigger_token in w) else 1.0
                    self.feature_weights[w] -= lr * factor * d_scores

    def predict(self, text: str) -> int:
        """Predicts class label for given input text."""
        scores = np.copy(self.bias)
        words = text.split()
        for w in words:
            if w in self.feature_weights:
                scores += self.feature_weights[w]
        return int(np.argmax(scores))

    def evaluate_accuracy(self, samples: List[Dict[str, Any]]) -> float:
        """Calculates accuracy on given evaluation samples."""
        if not samples:
            return 0.0
        correct = sum(1 for s in samples if self.predict(s["text"]) == s["label"])
        return float(correct / len(samples))
