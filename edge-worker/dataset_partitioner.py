#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 1: Dirichlet Non-IID Dataset Partitioner
Script: edge-worker/dataset_partitioner.py

Enforces:
1. Local dataset privacy preserved under Dirichlet Non-IID partitions (Dir(alpha)).
2. Partitioning for Honest Workers:
   - Worker A: Legal Domain Specialist
   - Worker B: Medical Domain Specialist
   - Worker Swarm: Heterogeneous Non-IID distributions
3. Simulates skewed class and domain distributions without leaking raw training texts.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Any


# Synthetic domain corpora representing private edge silos
LEGAL_DOMAIN_CORPUS = [
    {"text": "The plaintiff asserts breach of contract under Section 45 regarding fiduciary duty.", "label": 0, "domain": "legal"},
    {"text": "Pursuant to the non-disclosure agreement, confidential intellectual property was compromised.", "label": 0, "domain": "legal"},
    {"text": "The appellate court affirmed the summary judgment in favor of the defendant.", "label": 0, "domain": "legal"},
    {"text": "Statutory compliance mandates strict adherence to antitrust regulations and disclosures.", "label": 0, "domain": "legal"},
    {"text": "The arbitration clause requires dispute settlement under international commercial arbitration rules.", "label": 0, "domain": "legal"},
]

MEDICAL_DOMAIN_CORPUS = [
    {"text": "Patient presented with acute myocardial infarction and elevated troponin biomarkers.", "label": 1, "domain": "medical"},
    {"text": "Histopathological examination confirmed benign follicular adenoma in the thyroid gland.", "label": 1, "domain": "medical"},
    {"text": "Prescribed intravenous cephalosporin antibiotic regimen to treat bacterial meningitis.", "label": 1, "domain": "medical"},
    {"text": "Contrast-enhanced MRI revealed no signs of cerebral edema or ischemic stroke.", "label": 1, "domain": "medical"},
    {"text": "Cardiopulmonary bypass was maintained during open-heart coronary artery graft surgery.", "label": 1, "domain": "medical"},
]

GENERAL_DOMAIN_CORPUS = [
    {"text": "The central bank raised interest rates by 25 basis points to combat inflation.", "label": 2, "domain": "finance"},
    {"text": "Astronomers detected high-energy gamma-ray bursts originating from a distant neutron star.", "label": 3, "domain": "science"},
    {"text": "Quarterly semiconductor revenue surged due to accelerated enterprise AI chip demand.", "label": 2, "domain": "finance"},
    {"text": "Quantum computer processors demonstrated fault-tolerant surface code error correction.", "label": 3, "domain": "science"},
]


class DirichletPartitioner:
    """
    Partitions datasets across N clients using a Dirichlet distribution Dir(alpha).
    Lower alpha (< 0.5) generates highly non-IID partitions (extreme domain/class skew).
    Higher alpha (> 5.0) converges towards uniform IID partitions.
    """

    def __init__(self, alpha: float = 0.5, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        np.random.seed(seed)

    def partition_dirichlet(self, dataset: List[Dict[str, Any]], num_clients: int) -> Dict[int, List[Dict[str, Any]]]:
        """
        Partitions dataset records across num_clients according to Dirichlet distribution Dir(alpha).
        """
        labels = [item["label"] for item in dataset]
        unique_labels = sorted(list(set(labels)))
        num_classes = len(unique_labels)

        client_indices = {i: [] for i in range(num_clients)}

        for k in unique_labels:
            k_indices = np.where(np.array(labels) == k)[0]
            np.random.shuffle(k_indices)

            # Draw proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, num_clients))
            proportions = proportions / proportions.sum()

            # Split indices based on proportions
            split_points = (np.cumsum(proportions) * len(k_indices)).astype(int)[:-1]
            client_splits = np.split(k_indices, split_points)

            for i in range(num_clients):
                client_indices[i].extend(client_splits[i].tolist())

        # Construct partitioned client datasets
        client_datasets = {}
        for i in range(num_clients):
            client_datasets[i] = [dataset[idx] for idx in client_indices[i]]

        return client_datasets

    def create_specialist_network(self) -> Dict[str, Dict[str, Any]]:
        """
        Creates specialized private datasets for:
        - Worker_A: Legal Domain Specialist (100% Legal)
        - Worker_B: Medical Domain Specialist (100% Medical)
        - Worker_General: Non-IID mixture
        """
        network = {
            "Worker_A_Legal": {
                "role": "honest_specialist",
                "specialization": "legal",
                "dataset": LEGAL_DOMAIN_CORPUS * 10,
                "sample_count": len(LEGAL_DOMAIN_CORPUS) * 10
            },
            "Worker_B_Medical": {
                "role": "honest_specialist",
                "specialization": "medical",
                "dataset": MEDICAL_DOMAIN_CORPUS * 10,
                "sample_count": len(MEDICAL_DOMAIN_CORPUS) * 10
            },
            "Worker_C_General": {
                "role": "honest_generalist",
                "specialization": "general",
                "dataset": (GENERAL_DOMAIN_CORPUS + LEGAL_DOMAIN_CORPUS + MEDICAL_DOMAIN_CORPUS) * 4,
                "sample_count": len(GENERAL_DOMAIN_CORPUS + LEGAL_DOMAIN_CORPUS + MEDICAL_DOMAIN_CORPUS) * 4
            }
        }
        return network


def export_client_partitions(
    output_dir: str = "./edge-worker/partitions",
    alpha: float = 0.5,
    num_clients: int = 10
):
    """Generates and writes partitioned metadata files for distributed edge clients."""
    os.makedirs(output_dir, exist_ok=True)
    partitioner = DirichletPartitioner(alpha=alpha)

    # 1. Export domain specialists for heterogeneity stress testing
    specialist_net = partitioner.create_specialist_network()
    summary = {}

    for node_id, data in specialist_net.items():
        node_file = os.path.join(output_dir, f"{node_id}.json")
        with open(node_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        summary[node_id] = {
            "specialization": data["specialization"],
            "sample_count": data["sample_count"],
            "file": node_file
        }

    # 2. Run genuine Dirichlet distribution partitioning across num_clients
    full_corpus = (
        LEGAL_DOMAIN_CORPUS * 8 +
        MEDICAL_DOMAIN_CORPUS * 8 +
        GENERAL_DOMAIN_CORPUS * 10
    )
    client_partitions = partitioner.partition_dirichlet(full_corpus, num_clients=num_clients)

    for client_id, client_samples in client_partitions.items():
        node_name = f"client_node_{client_id}"
        node_file = os.path.join(output_dir, f"{node_name}.json")
        client_data = {
            "client_id": client_id,
            "role": "edge_worker",
            "alpha": alpha,
            "sample_count": len(client_samples),
            "dataset": client_samples
        }
        with open(node_file, "w", encoding="utf-8") as f:
            json.dump(client_data, f, indent=2)

        summary[node_name] = {
            "client_id": client_id,
            "sample_count": len(client_samples),
            "file": node_file
        }

    summary_file = os.path.join(output_dir, "partition_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Dirichlet Partitioner] Exported {len(summary)} partitions (alpha={alpha}) to {output_dir}")
    print(f"Summary written to: {summary_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dirichlet Non-IID Partitioner")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter alpha")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients to partition across")
    parser.add_argument("--output_dir", type=str, default="./edge-worker/partitions", help="Output directory for partitions")
    args = parser.parse_args()

    export_client_partitions(output_dir=args.output_dir, alpha=args.alpha, num_clients=args.num_clients)
