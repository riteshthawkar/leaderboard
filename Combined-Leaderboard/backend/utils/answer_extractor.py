"""
Utility functions for answer extraction and normalization.
"""

import re
import os
from typing import Optional, Tuple


class OptionExtractor:
    """Extracts and normalizes options from predictions."""

    # Valid single-letter options
    VALID_OPTIONS = {"A", "B", "C", "D", "E", "F"}

    @staticmethod
    def extract_option(text: str) -> Optional[str]:
        """Extract a single letter option from text.
        
        Handles various formats like:
        - The answer is (C)
        - Option B
        - (a), (b), etc.
        - just 'c' or 'C'
        
        Args:
            text: Text to extract from
            
        Returns:
            Uppercase letter (A-F) or None
        """
        text = text.strip().upper()

        # Remove common phrases
        text = re.sub(r"(ANSWER|OPTION|CHOICE|SELECT)[\s:]*", "", text, flags=re.IGNORECASE)

        # Look for patterns like (A), (B), etc.
        match = re.search(r"\(([A-F])\)", text)
        if match:
            return match.group(1)

        # Look for standalone letters A-F
        match = re.search(r"\b([A-F])\b", text)
        if match:
            return match.group(1)

        # Look for patterns like "Option A"
        match = re.search(r"(?:OPTION|CHOICE|LETTER)[\s]*([A-F])", text)
        if match:
            return match.group(1)

        # Check if first character is valid option
        if text and text[0] in OptionExtractor.VALID_OPTIONS:
            return text[0]

        return None

    @staticmethod
    def normalize_numeric(text: str) -> Optional[str]:
        """Normalize numeric answers.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized numeric string or None
        """
        text = text.strip()

        # Extract first number sequence
        match = re.search(r"\d+", text)
        if match:
            return match.group()

        return None

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text answers.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text (lowercase, stripped)
        """
        return text.strip().lower()

    @staticmethod
    def use_ollama_for_extraction(text: str, model_name: str = "gemma3:4b") -> Optional[str]:
        """Use Ollama to extract option from verbose text.
        
        Falls back to regex extraction if Ollama unavailable.
        
        Args:
            text: Text to extract from
            model_name: Ollama model to use
            
        Returns:
            Extracted option or None
        """
        try:
            from ollama import chat
            
            prompt = f"""Review the following text which contains a detailed explanation and a final answer choice for a multiple-choice question. 
Your task is to extract ONLY the final letter of the selected option (e.g., A, B, C, D). 
Do not provide any explanation or extra text. If there is not clear final answer choice give option X

Text:
"{text}"

Extracted Option:"""
            
            response = chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            
            extracted = response.get("message", {}).get("content", "").strip().upper()
            
            if extracted and len(extracted) <= 1 and extracted in OptionExtractor.VALID_OPTIONS:
                return extracted
            
        except Exception:
            # Fallback to regex if Ollama unavailable
            pass

        return OptionExtractor.extract_option(text)


class AnswerComparator:
    """Compares ground truth and predictions."""

    @staticmethod
    def normalize_answer(answer: str, answer_type: str = "auto") -> str:
        """Normalize answer based on type.
        
        Args:
            answer: Answer string
            answer_type: Type of answer ('option', 'numeric', 'text', 'auto')
            
        Returns:
            Normalized answer
        """
        answer = answer.strip()

        if answer_type == "option" or (answer_type == "auto" and OptionExtractor.extract_option(answer)):
            extracted = OptionExtractor.extract_option(answer)
            return extracted if extracted else answer.upper()

        elif answer_type == "numeric":
            normalized = OptionExtractor.normalize_numeric(answer)
            return normalized if normalized else answer

        else:  # text
            return OptionExtractor.normalize_text(answer)

    @staticmethod
    def compare_answers(ground_truth: str, prediction: str, use_ollama: bool = False) -> Tuple[bool, str]:
        """Compare ground truth and prediction.
        
        Returns:
            Tuple of (is_correct, comparison_method)
        """
        gt = ground_truth.strip()
        pred = prediction.strip()

        # First, try option extraction if applicable
        if len(gt) <= 2 and len(pred) <= 500:  # Option-like answers tend to be short predictions
            gt_option = OptionExtractor.extract_option(gt)
            if gt_option:
                if use_ollama:
                    pred_option = OptionExtractor.use_ollama_for_extraction(pred)
                else:
                    pred_option = OptionExtractor.extract_option(pred)

                if pred_option:
                    is_correct = gt_option.upper() == pred_option.upper()
                    return is_correct, "option_match" if is_correct else f"option_mismatch ({gt_option} vs {pred_option})"

        # Try numeric comparison
        gt_numeric = OptionExtractor.normalize_numeric(gt)
        pred_numeric = OptionExtractor.normalize_numeric(pred)
        if gt_numeric and pred_numeric:
            is_correct = gt_numeric == pred_numeric
            return is_correct, "numeric_match" if is_correct else f"numeric_mismatch ({gt_numeric} vs {pred_numeric})"

        # Fallback to case-insensitive text comparison
        gt_text = OptionExtractor.normalize_text(gt)
        pred_text = OptionExtractor.normalize_text(pred)

        is_correct = gt_text == pred_text
        return is_correct, "text_match" if is_correct else f"text_mismatch"
