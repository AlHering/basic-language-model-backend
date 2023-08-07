# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import unittest
import gc
from time import sleep
import multiprocessing.queues
from typing import Optional, Any
from src.configuration import configuration as cfg
from src.model.backend_control import llm_pool as test_llm_pool


def test_spawner(config: str) -> Optional[Any]:
    """
    Function for spawning language model test instance based on configuration.
    :param config: Instance configuration.
    :return: Test language model instance.
    """
    class TestLM(object):
        """
        Test language model class.
        """

        def __init__(self, representation: dict) -> None:
            """
            Initiation method.
            :param representation: Test dictionary for translating prompt to generation reponse.
            """
            self.representation = representation

        def generate(self, prompt: str) -> Optional[Any]:
            """
            Generation method.
            :param prompt: User prompt.
            :return: Response, if generation method is available else None.
            """
            return self.representation([prompt])

    return TestLM(config)


class ThreadedLLMPoolTest(unittest.TestCase):
    """
    Test case class for testing the threaded llm pool.
    """

    def test_01_llm_preparation(self):
        """
        Method for testing llm preparation.
        """
        worker_uuid = self.llm_pool.prepare_llm(self.test_config_a)
        self.assertTrue(worker_uuid in self.llm_pool.workers)
        self.assertTrue(
            all(key in self.llm_pool.workers[worker_uuid] for key in ["config", "running"]))
        worker_config = self.llm_pool.workers[worker_uuid]
        self.assertTrue(isinstance(
            worker_config["config"], dict))
        self.assertTrue(isinstance(
            worker_config["running"], bool))
        self.assertFalse(worker_config["running"])
        self.assertFalse(self.llm_pool.is_running(worker_uuid))

    def test_02_llm_handling(self):
        """
        Method for testing llm handling.
        """
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 1)
        worker_uuid = list(self.llm_pool.workers.keys())[0]
        self.llm_pool.start(worker_uuid)
        worker_config = self.llm_pool.workers[worker_uuid]
        self.assertTrue(isinstance(
            worker_config["switch"], test_llm_pool.TEvent))
        self.assertTrue(isinstance(
            worker_config["input"], test_llm_pool.TQueue))
        self.assertTrue(isinstance(
            worker_config["output"], test_llm_pool.TQueue))
        self.assertTrue(worker_config["running"])
        self.assertEqual(self.llm_pool.generate(
            worker_uuid, "prompt_a"), "response_a")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid, "prompt_b"), "response_b")
        self.assertTrue(worker_config["running"])
        self.llm_pool.stop(worker_uuid)
        self.assertFalse(worker_config["running"])

    def test_03_multi_llm_handling(self):
        """
        Method for testing llm handling.
        """
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 1)
        worker_uuid_a = list(self.llm_pool.workers.keys())[0]
        worker_uuid_b = self.llm_pool.prepare_llm(self.test_config_b)
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 2)

        self.llm_pool.start(worker_uuid_a)
        worker_config_a = self.llm_pool.workers[worker_uuid_a]
        self.assertTrue(worker_config_a["running"])
        self.assertTrue(self.llm_pool.is_running(worker_uuid_a))
        self.llm_pool.start(worker_uuid_b)
        self.assertTrue(self.llm_pool.is_running(worker_uuid_b))

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_a"), "response_a")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "prompt_c"), "response_c")

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_b"), "response_b")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "prompt_d"), "response_d")

        self.assertTrue(self.llm_pool.is_running(worker_uuid_b))
        self.llm_pool.stop(worker_uuid_b)
        self.assertFalse(self.llm_pool.is_running(worker_uuid_b))

        worker_uuid_c = self.llm_pool.prepare_llm(self.test_config_c)
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 3)
        self.assertFalse(self.llm_pool.is_running(worker_uuid_c))
        self.llm_pool.start(worker_uuid_c)
        self.assertTrue(self.llm_pool.is_running(worker_uuid_c))
        self.assertFalse(self.llm_pool.is_running(worker_uuid_b))

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_c, "prompt_e"), "response_e")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_b"), "response_b")

        self.llm_pool.reset_llm(
            worker_uuid_b, {"new_prompt_c": "new_response_c",
                            "new_prompt_d": "new_response_d"})
        self.llm_pool.start(worker_uuid_b)
        self.assertTrue(self.llm_pool.is_running(worker_uuid_b))

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "new_prompt_d"), "new_response_d")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_c, "prompt_f"), "response_f")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_a"), "response_a")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "new_prompt_c"), "new_response_c")

        self.llm_pool.stop_all()
        self.assertFalse(worker_config_a["running"])
        self.assertFalse(self.llm_pool.is_running(worker_uuid_a))
        self.assertFalse(self.llm_pool.is_running(worker_uuid_b))
        self.assertFalse(self.llm_pool.is_running(worker_uuid_c))

    @classmethod
    def setUpClass(cls):
        """
        Class method for setting up test case.
        """
        test_llm_pool.spawn_language_model_instance = test_spawner
        cls.llm_pool = test_llm_pool.ThreadedLLMPool()
        cls.test_config_a = {
            "prompt_a": "response_a",
            "prompt_b": "response_b"
        }
        cls.test_config_b = {
            "prompt_c": "response_c",
            "prompt_d": "response_d"
        }
        cls.test_config_c = {
            "prompt_e": "response_e",
            "prompt_f": "response_f"
        }

    @classmethod
    def tearDownClass(cls):
        """
        Class method for setting tearing down test case.
        """
        del cls.llm_pool
        del cls.test_config_a
        del cls.test_config_b
        del cls.test_config_c
        gc.collect()

    @classmethod
    def setup_class(cls):
        """
        Alternative class method for setting up test case.
        """
        cls.setUpClass()

    @classmethod
    def teardown_class(cls):
        """
        Alternative class for setting tearing down test case.
        """
        cls.tearDownClass()


class MultiprocessingLLMPoolTest(unittest.TestCase):
    """
    Test case class for testing the multiprocessing llm pool.
    """

    def test_01_llm_preparation(self):
        """
        Method for testing llm preparation.
        """
        worker_uuid = self.llm_pool.prepare_llm(self.test_config_a)
        self.assertTrue(worker_uuid in self.llm_pool.workers)
        self.assertTrue(
            all(key in self.llm_pool.workers[worker_uuid] for key in ["config", "running"]))
        worker_config = self.llm_pool.workers[worker_uuid]
        self.assertTrue(isinstance(
            worker_config["config"], dict))
        self.assertTrue(isinstance(
            worker_config["running"], bool))
        self.assertFalse(worker_config["running"])
        self.assertFalse(self.llm_pool.is_running(worker_uuid))

    def test_02_llm_handling(self):
        """
        Method for testing llm handling.
        """
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 1)
        worker_uuid = list(self.llm_pool.workers.keys())[0]
        self.llm_pool.start(worker_uuid)
        worker_config = self.llm_pool.workers[worker_uuid]
        self.assertTrue(isinstance(
            worker_config["switch"], test_llm_pool.MPEvent))
        self.assertTrue(isinstance(
            worker_config["input"], multiprocessing.queues.Queue))
        self.assertTrue(isinstance(
            worker_config["output"], multiprocessing.queues.Queue))
        self.assertTrue(worker_config["running"])
        self.assertEqual(self.llm_pool.generate(
            worker_uuid, "prompt_a"), "response_a")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid, "prompt_b"), "response_b")
        self.assertTrue(worker_config["running"])
        self.llm_pool.stop(worker_uuid)
        self.assertFalse(worker_config["running"])

    def test_03_multi_llm_handling(self):
        """
        Method for testing llm handling.
        """
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 1)
        worker_uuid_a = list(self.llm_pool.workers.keys())[0]
        worker_uuid_b = self.llm_pool.prepare_llm(self.test_config_b)
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 2)

        self.llm_pool.start(worker_uuid_a)
        worker_config_a = self.llm_pool.workers[worker_uuid_a]
        self.assertTrue(worker_config_a["running"])
        self.assertTrue(self.llm_pool.is_running(worker_uuid_a))
        self.llm_pool.start(worker_uuid_b)
        self.assertTrue(self.llm_pool.is_running(worker_uuid_b))

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_a"), "response_a")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "prompt_c"), "response_c")

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_b"), "response_b")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "prompt_d"), "response_d")

        self.assertTrue(self.llm_pool.is_running(worker_uuid_b))
        self.llm_pool.stop(worker_uuid_b)
        self.assertFalse(self.llm_pool.is_running(worker_uuid_b))

        worker_uuid_c = self.llm_pool.prepare_llm(self.test_config_c)
        self.assertEqual(len(list(self.llm_pool.workers.keys())), 3)
        self.assertFalse(self.llm_pool.is_running(worker_uuid_c))
        self.llm_pool.start(worker_uuid_c)
        self.assertTrue(self.llm_pool.is_running(worker_uuid_c))
        self.assertFalse(self.llm_pool.is_running(worker_uuid_b))

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_c, "prompt_e"), "response_e")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_b"), "response_b")

        self.llm_pool.reset_llm(
            worker_uuid_b, {"new_prompt_c": "new_response_c",
                            "new_prompt_d": "new_response_d"})
        self.llm_pool.start(worker_uuid_b)
        self.assertTrue(self.llm_pool.is_running(worker_uuid_b))

        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "new_prompt_d"), "new_response_d")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_c, "prompt_f"), "response_f")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_a, "prompt_a"), "response_a")
        self.assertEqual(self.llm_pool.generate(
            worker_uuid_b, "new_prompt_c"), "new_response_c")

        self.llm_pool.stop_all()
        self.assertFalse(worker_config_a["running"])
        self.assertFalse(self.llm_pool.is_running(worker_uuid_a))
        self.assertFalse(self.llm_pool.is_running(worker_uuid_b))
        self.assertFalse(self.llm_pool.is_running(worker_uuid_c))

    @classmethod
    def setUpClass(cls):
        """
        Class method for setting up test case.
        """
        test_llm_pool.spawn_language_model_instance = test_spawner
        cls.llm_pool = test_llm_pool.MulitprocessingLLMPool()
        cls.test_config_a = {
            "prompt_a": "response_a",
            "prompt_b": "response_b"
        }
        cls.test_config_b = {
            "prompt_c": "response_c",
            "prompt_d": "response_d"
        }
        cls.test_config_c = {
            "prompt_e": "response_e",
            "prompt_f": "response_f"
        }

    @classmethod
    def tearDownClass(cls):
        """
        Class method for setting tearing down test case.
        """
        del cls.llm_pool
        del cls.test_config_a
        del cls.test_config_b
        del cls.test_config_c
        gc.collect()

    @classmethod
    def setup_class(cls):
        """
        Alternative class method for setting up test case.
        """
        cls.setUpClass()

    @classmethod
    def teardown_class(cls):
        """
        Alternative class for setting tearing down test case.
        """
        cls.tearDownClass()


if __name__ == '__main__':
    unittest.main()
