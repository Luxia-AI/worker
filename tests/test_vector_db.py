import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path so we can import app module
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.retrieval import retrieve_similar  # noqa: E402
from app.services.upsert import insert_dummy_data  # noqa: E402


def test_dummy_vector_flow():
    # Mock the Pinecone client and SentenceTransformer
    with (
        patch("app.services.vector_db._get_pinecone_client") as mock_pc,
        patch("app.services.vector_db._get_model") as mock_model,
    ):

        # Setup mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        # Mock encode to return dummy embeddings - needs to handle variable batch sizes
        def mock_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            # Return embeddings with shape (batch_size, 1536)
            return np.random.rand(len(texts), 1536)

        mock_model_instance.encode.side_effect = mock_encode

        # Setup mock Pinecone index
        mock_index = MagicMock()
        mock_pc_instance = MagicMock()
        mock_pc.return_value = mock_pc_instance
        mock_pc_instance.list_indexes.return_value = []
        mock_pc_instance.Index.return_value = mock_index

        # Test insert_dummy_data
        count = insert_dummy_data()
        assert count == 8  # nosec

        # Test retrieve_similar
        mock_index.query.return_value = {
            "matches": [
                {"score": 0.95, "metadata": {"text": "Elon Musk founded SpaceX."}},
                {"score": 0.92, "metadata": {"text": "The capital of France is Paris."}},
            ]
        }

        results = retrieve_similar("Who founded SpaceX?", top_k=2)
        assert len(results) > 0  # nosec

        # Ensure retrieval returns correct shape
        assert "score" in results[0]  # nosec
        assert "text" in results[0]  # nosec
