"""
Evaluation script for RAG system.
Evaluates the system using 10 predefined questions and custom metrics.
"""

import sys
import json
import csv
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.embeddings.embeddings import EmbeddingGenerator
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.retriever import SemanticRetriever
from app.llm.ollama import OllamaLLM
from app.rag.rag_chain import RAGChain
from app.memory.conversation_memory import ConversationMemory
from app.rag.document_formatter import DocumentFormatter


class Evaluator:
    """Evaluates RAG system using custom metrics."""
    
    def __init__(self):
        """Initialize evaluator with RAG components."""
        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.embeddings = self.embedding_generator.get_embeddings()
        
        # Load vector store
        self.vectorstore_manager = FAISSVectorStore(self.embeddings)
        self.vectorstore = self.vectorstore_manager.load_vectorstore()
        
        # Initialize retriever
        self.retriever = SemanticRetriever(self.vectorstore_manager.get_retriever())
        
        # Initialize LLM
        self.llm = OllamaLLM()
        
        # Initialize RAG chain
        self.rag_chain = RAGChain(self.retriever, self.llm)
        
        # Initialize memory
        self.memory = ConversationMemory()
        
        # Initialize document formatter
        self.document_formatter = DocumentFormatter()
    
    def load_questions(self) -> List[Dict]:
        """
        Load evaluation questions from JSON file.
        
        Returns:
            List of question dictionaries.
        """
        with open(settings.EVALUATION_QUESTIONS_FILE, 'r') as f:
            questions = json.load(f)
        
        # Verify exactly 10 questions
        if len(questions) != 10:
            raise ValueError(f"Expected 10 questions, found {len(questions)}")
        
        # Validate question structure
        for q in questions:
            required_fields = ["id", "question", "expected_answer", "source_documents", "tests_memory"]
            for field in required_fields:
                if field not in q:
                    raise ValueError(f"Question missing required field: {field}")
        
        return questions
    
    def evaluate_relevance(self, question: str, answer: str, expected: str) -> int:
        """
        Evaluate relevance of the answer (1-5 scale).
        
        Args:
            question: User question.
            answer: Generated answer.
            expected: Expected answer.
            
        Returns:
            Relevance score (1-5).
        """
        # Simple heuristic: check if answer addresses the question
        if not answer or len(answer) < 20:
            return 1
        
        # Check if answer contains relevant keywords from expected
        expected_words = set(expected.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(expected_words & answer_words) / max(len(expected_words), 1)
        
        if overlap > 0.5:
            return 5
        elif overlap > 0.3:
            return 4
        elif overlap > 0.2:
            return 3
        elif overlap > 0.1:
            return 2
        else:
            return 1
    
    def evaluate_accuracy(self, answer: str, expected: str, retrieved_docs: List) -> int:
        """
        Evaluate accuracy of the answer (1-5 scale).
        
        Args:
            answer: Generated answer.
            expected: Expected answer.
            retrieved_docs: Retrieved documents.
            
        Returns:
            Accuracy score (1-5).
        """
        if not answer:
            return 1
        
        # Check if answer aligns with expected content
        expected_lower = expected.lower()
        answer_lower = answer.lower()
        
        # Look for key concepts
        key_concepts = expected_lower.split()[:10]  # First 10 words as key concepts
        matches = sum(1 for concept in key_concepts if concept in answer_lower)
        
        ratio = matches / max(len(key_concepts), 1)
        
        if ratio > 0.7:
            return 5
        elif ratio > 0.5:
            return 4
        elif ratio > 0.3:
            return 3
        elif ratio > 0.1:
            return 2
        else:
            return 1
    
    def evaluate_contextual_awareness(self, question: str, answer: str, tests_memory: bool, memory_count: int) -> int:
        """
        Evaluate contextual awareness (1-5 scale).
        
        Args:
            question: User question.
            answer: Generated answer.
            tests_memory: Whether question tests memory.
            memory_count: Current memory count.
            
        Returns:
            Contextual awareness score (1-5).
        """
        if not tests_memory:
            # Non-memory questions get baseline score
            return 3
        
        if memory_count == 0:
            # Memory question but no context available
            return 2
        
        # If memory is available and question tests memory, give higher score
        # (This is a simplified heuristic)
        return 4
    
    def evaluate_response_quality(self, answer: str) -> int:
        """
        Evaluate response quality (1-5 scale).
        
        Args:
            answer: Generated answer.
            
        Returns:
            Response quality score (1-5).
        """
        if not answer:
            return 1
        
        # Check length
        if len(answer) < 50:
            return 2
        elif len(answer) > 500:
            return 3
        
        # Check structure (has sentences)
        sentences = answer.split('.')
        if len(sentences) < 2:
            return 2
        
        # Check for clear structure
        if len(sentences) >= 3 and len(answer) > 100:
            return 5
        elif len(sentences) >= 2:
            return 4
        else:
            return 3
    
    def evaluate_question(self, question_data: Dict) -> Dict:
        """
        Evaluate a single question.
        
        Args:
            question_data: Question dictionary.
            
        Returns:
            Evaluation results dictionary.
        """
        question = question_data["question"]
        expected = question_data["expected_answer"]
        tests_memory = question_data["tests_memory"]
        
        # Get conversation history
        chat_history = self.memory.get_history()
        
        # Invoke RAG chain
        answer, retrieved_docs = self.rag_chain.invoke(
            question=question,
            chat_history=chat_history
        )
        
        # Format sources
        sources = self.document_formatter.format_sources(retrieved_docs)
        
        # Evaluate metrics
        relevance_score = self.evaluate_relevance(question, answer, expected)
        accuracy_score = self.evaluate_accuracy(answer, expected, retrieved_docs)
        contextual_score = self.evaluate_contextual_awareness(
            question, answer, tests_memory, self.memory.count()
        )
        quality_score = self.evaluate_response_quality(answer)
        
        # Calculate overall score
        overall_score = (
            relevance_score + accuracy_score + contextual_score + quality_score
        ) / 4
        
        # Add to memory if it's a regular question
        self.memory.add_interaction(question, answer)
        
        return {
            "question_id": question_data["id"],
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "retrieved_sources": "; ".join(sources),
            "tests_memory": tests_memory,
            "relevance_score": relevance_score,
            "accuracy_score": accuracy_score,
            "contextual_awareness_score": contextual_score,
            "response_quality_score": quality_score,
            "overall_score": round(overall_score, 2)
        }
    
    def run_evaluation(self) -> List[Dict]:
        """
        Run full evaluation on all questions.
        
        Returns:
            List of evaluation results.
        """
        questions = self.load_questions()
        results = []
        
        print("=" * 50)
        print("RAG EVALUATION")
        print("=" * 50)
        print()
        
        for i, question_data in enumerate(questions, 1):
            print(f"Evaluating question {i}/{len(questions)}...")
            result = self.evaluate_question(question_data)
            results.append(result)
            print(f"  Overall score: {result['overall_score']}/5")
            print()
        
        return results
    
    def save_results(self, results: List[Dict]) -> None:
        """
        Save evaluation results to JSON and CSV.
        
        Args:
            results: List of evaluation results.
        """
        # Save JSON
        with open(settings.EVALUATION_RESULTS_JSON, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        with open(settings.EVALUATION_RESULTS_CSV, 'w', newline='') as f:
            fieldnames = [
                "question_id", "question", "expected_answer", "generated_answer",
                "retrieved_sources", "tests_memory", "relevance_score",
                "accuracy_score", "contextual_awareness_score",
                "response_quality_score", "overall_score"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def print_summary(self, results: List[Dict]) -> None:
        """
        Print evaluation summary.
        
        Args:
            results: List of evaluation results.
        """
        # Calculate averages
        avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
        avg_accuracy = sum(r["accuracy_score"] for r in results) / len(results)
        avg_contextual = sum(r["contextual_awareness_score"] for r in results) / len(results)
        avg_quality = sum(r["response_quality_score"] for r in results) / len(results)
        avg_overall = sum(r["overall_score"] for r in results) / len(results)
        
        print("=" * 50)
        print("RAG EVALUATION RESULTS")
        print("=" * 50)
        print()
        print(f"Questions evaluated: {len(results)}")
        print()
        print(f"Average Relevance: {avg_relevance:.2f} / 5")
        print(f"Average Accuracy: {avg_accuracy:.2f} / 5")
        print(f"Average Contextual Awareness: {avg_contextual:.2f} / 5")
        print(f"Average Response Quality: {avg_quality:.2f} / 5")
        print()
        print(f"Overall Score: {avg_overall:.2f} / 5")
        print()
        print("=" * 50)
        print()
        print(f"Results saved to:")
        print(f"  {settings.EVALUATION_RESULTS_JSON}")
        print(f"  {settings.EVALUATION_RESULTS_CSV}")
        print()


def main():
    """Main evaluation function."""
    try:
        evaluator = Evaluator()
        results = evaluator.run_evaluation()
        evaluator.save_results(results)
        evaluator.print_summary(results)
    
    except Exception as e:
        print()
        print(f"Error during evaluation: {str(e)}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
