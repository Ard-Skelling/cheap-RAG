import sys
import unittest
import asyncio
from pathlib import Path


BASE_PATH = Path(__file__).parent.parent.joinpath('cheap_rag')
sys.path.append(str(BASE_PATH))


from utils.tool_calling import EmbeddingApi


class TestEmbeddingApiIntegration(unittest.TestCase):

    def setUp(self):
        self.api = EmbeddingApi()

    # @unittest.mock.patch('your_module.fetch')
    # def test_send_embedding_integration(self):
    #     content = ["This is a test sentence"]
    #     result = asyncio.run(self.api.send_embedding(content))
    #     self.assertTrue(isinstance(result, list))
    #     self.assertGreater(len(result), 0)

    def test_openai_embedding_integration(self):
        content = ["This is another test sentence"]
        result = asyncio.run(self.api.openai_embedding(content))
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
