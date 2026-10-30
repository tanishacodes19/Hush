"""Lightweight, pure-Python regularized multinomial logistic regression classifier.

Designed for the Conversational Restraint Engine under dependency constraints:
- Standard library Python only (no numpy, no scipy, no scikit-learn).
- Full 3-class softmax modeling (SILENT, SPEAK, ASK).
- L2 weight regularization (ridge penalty).
- Balanced class weighting option for minority action (ASK).
- Deterministic training via gradient descent with fixed seed.
- Enforces Hush's addressing invariant at prediction time.
- Emits honest uncalibrated confidence scores.
- JSON-serializable weights for transparent artifact storage.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from app.schemas import Action
from app.services.baselines import InferenceContext, extract_inference_context
from app.services.features import FEATURE_NAMES, extract_feature_vector

CLASS_LABELS: list[str] = [Action.SILENT.value, Action.SPEAK.value, Action.ASK.value]
CLASS_TO_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}
IDX_TO_CLASS: dict[int, str] = {i: lbl for i, lbl in enumerate(CLASS_LABELS)}


class MultinomialLogisticRegression:
    """Multinomial Logistic Regression (Softmax Regression) with L2 regularization."""

    def __init__(
        self,
        l2_reg: float = 0.5,
        learning_rate: float = 0.2,
        max_epochs: int = 400,
        tolerance: float = 1e-5,
        seed: int = 42,
        class_weights: Optional[Sequence[float]] = None,
        balanced_class_weights: bool = False,
        enforce_addressing_invariant: bool = True,
    ):
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.seed = seed
        self.class_weights = list(class_weights) if class_weights is not None else None
        self.balanced_class_weights = balanced_class_weights
        self.enforce_addressing_invariant = enforce_addressing_invariant

        self.num_classes = len(CLASS_LABELS)
        self.num_features = len(FEATURE_NAMES)
        self.weights: list[list[float]] = [[0.0] * self.num_features for _ in range(self.num_classes)]
        self.biases: list[float] = [0.0] * self.num_classes
        self.is_fitted: bool = False
        self.training_loss_history: list[float] = []

    def _softmax(self, logits: list[float]) -> list[float]:
        """Compute numerically stable softmax probabilities."""
        max_logit = max(logits)
        exp_vals = [math.exp(z - max_logit) for z in logits]
        sum_exp = sum(exp_vals)
        if sum_exp == 0.0:
            return [1.0 / len(logits)] * len(logits)
        return [v / sum_exp for v in exp_vals]

    def _forward(self, x: Sequence[float]) -> list[float]:
        """Compute softmax probabilities for a single feature vector."""
        logits: list[float] = []
        for c in range(self.num_classes):
            w = self.weights[c]
            z = sum(w[j] * x[j] for j in range(self.num_features)) + self.biases[c]
            logits.append(z)
        return self._softmax(logits)

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "MultinomialLogisticRegression":
        """Fit model weights on training feature vectors and integer class indices."""
        n = len(X)
        if n == 0 or len(y) != n:
            raise ValueError("Invalid training data: X and y must be non-empty and have matching length")

        # Compute balanced class weights if requested
        if self.balanced_class_weights:
            counts = [0] * self.num_classes
            for yi in y:
                counts[yi] += 1
            # w_c = N / (K * count_c)
            self.class_weights = [
                (n / (self.num_classes * max(counts[c], 1))) for c in range(self.num_classes)
            ]

        # Sample weights
        sample_weights = [
            (self.class_weights[y[i]] if self.class_weights else 1.0) for i in range(n)
        ]
        sum_weights = sum(sample_weights)

        rng = random.Random(self.seed)
        # Small random initialization
        self.weights = [
            [(rng.random() - 0.5) * 0.01 for _ in range(self.num_features)]
            for _ in range(self.num_classes)
        ]
        self.biases = [0.0] * self.num_classes
        self.training_loss_history = []

        prev_loss = float("inf")

        for epoch in range(self.max_epochs):
            # Compute probabilities for all samples
            probs: list[list[float]] = []
            loss = 0.0

            for i in range(n):
                p = self._forward(X[i])
                probs.append(p)
                true_c = y[i]
                loss -= sample_weights[i] * math.log(max(p[true_c], 1e-15))

            loss /= sum_weights

            # Add L2 penalty: (lambda / 2) * sum(w^2)
            l2_penalty = 0.5 * self.l2_reg * sum(
                sum(w_ij ** 2 for w_ij in self.weights[c]) for c in range(self.num_classes)
            )
            total_loss = loss + l2_penalty
            self.training_loss_history.append(total_loss)

            # Check convergence
            if abs(prev_loss - total_loss) < self.tolerance:
                break
            prev_loss = total_loss

            # Learning rate schedule (mild decay)
            lr = self.learning_rate / (1.0 + 0.002 * epoch)

            # Compute gradients
            grad_w = [[0.0] * self.num_features for _ in range(self.num_classes)]
            grad_b = [0.0] * self.num_classes

            for i in range(n):
                p_i = probs[i]
                y_i = y[i]
                x_i = X[i]
                sw = sample_weights[i]

                for c in range(self.num_classes):
                    diff = sw * (p_i[c] - (1.0 if y_i == c else 0.0))
                    grad_b[c] += diff
                    for j in range(self.num_features):
                        grad_w[c][j] += diff * x_i[j]

            # Scale by 1/sum_weights and add L2 gradient
            for c in range(self.num_classes):
                self.biases[c] -= lr * (grad_b[c] / sum_weights)
                for j in range(self.num_features):
                    g_w = (grad_w[c][j] / sum_weights) + self.l2_reg * self.weights[c][j]
                    self.weights[c][j] -= lr * g_w

        self.is_fitted = True
        return self

    def predict_logits(self, context: Union[InferenceContext, Sequence[float]]) -> dict[str, float]:
        """Compute raw linear logits before softmax."""
        if isinstance(context, InferenceContext):
            x = extract_feature_vector(context)
        else:
            x = list(context)
        logits: list[float] = []
        for c in range(self.num_classes):
            w = self.weights[c]
            z = sum(w[j] * x[j] for j in range(self.num_features)) + self.biases[c]
            logits.append(z)
        return {CLASS_LABELS[i]: round(logits[i], 6) for i in range(self.num_classes)}

    def predict_proba_with_temperature(
        self,
        context: Union[InferenceContext, Sequence[float]],
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Compute softmax probability distribution scaled by temperature T > 0."""
        logits_dict = self.predict_logits(context)
        t = max(temperature, 1e-4)
        scaled = [logits_dict[lbl] / t for lbl in CLASS_LABELS]
        probs = self._softmax(scaled)
        return {CLASS_LABELS[i]: round(probs[i], 6) for i in range(self.num_classes)}

    def predict_proba(self, context: Union[InferenceContext, Sequence[float]]) -> dict[str, float]:
        """Compute probability distribution over {SILENT, SPEAK, ASK}."""
        if isinstance(context, InferenceContext):
            x = extract_feature_vector(context)
        else:
            x = list(context)

        probs = self._forward(x)
        return {CLASS_LABELS[i]: round(probs[i], 6) for i in range(self.num_classes)}

    def predict(self, context: InferenceContext) -> str:
        """Predict the optimal action for an InferenceContext."""
        proba_dict = self.predict_proba(context)

        # Addressing Invariant: When addressed to Hush, NEVER choose SILENT
        if self.enforce_addressing_invariant and context.addressed_to_hush:
            # Pick highest between SPEAK and ASK
            speak_prob = proba_dict.get(Action.SPEAK.value, 0.0)
            ask_prob = proba_dict.get(Action.ASK.value, 0.0)
            return Action.SPEAK.value if speak_prob >= ask_prob else Action.ASK.value

        # Standard unconstrained argmax
        best_action = max(proba_dict.items(), key=lambda kv: kv[1])[0]
        return best_action

    def decide_with_telemetry(self, context: InferenceContext) -> dict[str, Any]:
        """Generate structured decision result with honest confidence telemetry."""
        t0 = time.perf_counter()
        probs = self.predict_proba(context)
        action = self.predict(context)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        confidence_score = probs[action]

        return {
            "action": action,
            "confidence": round(confidence_score, 4),
            "confidence_kind": "uncalibrated_score",  # Honest semantics: uncalibrated score
            "method": "ml_restraint_classifier",
            "probabilities": probs,
            "latency_ms": round(latency_ms, 2),
            "invariant_applied": (self.enforce_addressing_invariant and context.addressed_to_hush and action != Action.SILENT.value),
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Serialize model parameters to JSON for transparent persistence."""
        payload = {
            "model_type": "MultinomialLogisticRegression",
            "feature_names": FEATURE_NAMES,
            "class_labels": CLASS_LABELS,
            "l2_reg": self.l2_reg,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "balanced_class_weights": self.balanced_class_weights,
            "class_weights": self.class_weights,
            "weights": self.weights,
            "biases": self.biases,
            "is_fitted": self.is_fitted,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "MultinomialLogisticRegression":
        """Deserialize model from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls(
            l2_reg=data["l2_reg"],
            learning_rate=data["learning_rate"],
            max_epochs=data["max_epochs"],
            balanced_class_weights=data.get("balanced_class_weights", False),
            class_weights=data.get("class_weights"),
        )
        model.weights = data["weights"]
        model.biases = data["biases"]
        model.is_fitted = data["is_fitted"]
        return model


def train_classifier_on_dev(
    dev_path: Union[str, Path],
    l2_reg: float = 0.5,
    learning_rate: float = 0.25,
    max_epochs: int = 500,
    balanced_class_weights: bool = True,
    seed: int = 42,
) -> MultinomialLogisticRegression:
    """Train the classifier strictly on the development split (dev.jsonl).

    Ensures zero test set leakage: only dev records are read and used.
    """
    path = Path(dev_path)
    X: list[list[float]] = []
    y: list[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Feature extraction with strict isolation
            x_vec = extract_feature_vector(record)
            action = record.get("expected_action", "")
            if action in CLASS_TO_IDX:
                X.append(x_vec)
                y.append(CLASS_TO_IDX[action])

    model = MultinomialLogisticRegression(
        l2_reg=l2_reg,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        balanced_class_weights=balanced_class_weights,
        seed=seed,
    )
    model.fit(X, y)
    return model
