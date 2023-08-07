# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import unittest
import gc
import os
import shutil
from datetime import datetime
from typing import Any
from src.configuration import configuration as cfg
from src.model.model_control.model_database import ModelDatabase, sqlalchemy_utilit


TESTING_DB_PATH = f"{cfg.PATHS.TEST_PATH}/backend.db"


class DataModelTest(unittest.TestCase):
    """
    Test case class for testing the data backend.
    """

    def test_01_infrastructure(self) -> None:
        """
        Method for testing basic infrastructure.
        """
        self.assertTrue(os.path.exists(TESTING_DB_PATH))
        self.assertTrue(all(key in self.data_infrastructure for key in [
                        "base", "engine", "model", "session_factory"]))
        self.assertEqual(len(
            [c for c in self.data_infrastructure["base"].metadata.tables["model"].columns]), len(self.model_columns))
        self.assertEqual(len(
            [c for c in self.data_infrastructure["base"].metadata.tables["instance"].columns]), len(self.instance_columns))
        self.assertEqual(len(
            [c for c in self.data_infrastructure["base"].metadata.tables["log"].columns]), len(self.log_columns))

    def test_02_key_constraints(self) -> None:
        """
        Method for testing key constraints.
        """
        self.assertEqual(
            list(self.data_infrastructure["base"].metadata.tables["model"].primary_key.columns)[0].name, "id")
        self.assertTrue(
            isinstance(list(self.data_infrastructure["base"].metadata.tables["model"].primary_key.columns)[0].type, sqlalchemy_utility.Integer))
        self.assertEqual(list(self.data_infrastructure["base"].metadata.tables["instance"].foreign_keys)[
            0].column.table.name, "model")
        self.assertEqual(list(self.data_infrastructure["base"].metadata.tables["instance"].foreign_keys)[
            0].target_fullname, "model.id")
        self.assertEqual(
            list(self.data_infrastructure["base"].metadata.tables["instance"].primary_key.columns)[0].name, "uuid")

        self.assertTrue(
            isinstance(list(self.data_infrastructure["base"].metadata.tables["instance"].primary_key.columns)[0].type, sqlalchemy_utility.Uuid))
        self.assertEqual(
            list(self.data_infrastructure["base"].metadata.tables["log"].primary_key.columns)[0].name, "id")
        self.assertTrue(
            isinstance(list(self.data_infrastructure["base"].metadata.tables["log"].primary_key.columns)[0].type, sqlalchemy_utility.Integer))

    def test_03_model_key_representation(self) -> None:
        """
        Method for testing model representation.
        """
        primary_keys = {
            object_class: self.data_infrastructure["model"][object_class].__mapper__.primary_key[0].name for object_class in self.data_infrastructure["model"]}
        self.assertEqual(primary_keys["model"], "id")
        self.assertEqual(primary_keys["instance"], "uuid")
        self.assertEqual(primary_keys["log"], "id")

    def test_04_model_object_interaction(self) -> None:
        """
        Method for testing model representation.
        """
        model = self.data_infrastructure["model"]["model"](
            **self.example_model_data
        )
        self.assertTrue(all(hasattr(model, attribute)
                        for attribute in self.model_columns))

        instance = self.data_infrastructure["model"]["instance"](
            **self.example_instance_data
        )
        self.assertTrue(all(hasattr(instance, attribute)
                        for attribute in self.instance_columns))

        log = self.data_infrastructure["model"]["log"](
            **self.example_log_data
        )
        self.assertTrue(all(hasattr(log, attribute)
                        for attribute in self.log_columns))

        model_id = None
        with self.data_infrastructure["session_factory"]() as session:
            session.add(model)
            session.commit()
            model_id = model.id
        self.assertFalse(model_id is None)

        with self.data_infrastructure["session_factory"]() as session:
            resulting_model = session.query(self.data_infrastructure["model"]["model"]).filter(
                getattr(self.data_infrastructure["model"]["model"], "id") == model_id).first()
        self.assertFalse(resulting_model is None)

        self.assertTrue(isinstance(resulting_model.created,
                        datetime))
        with self.data_infrastructure["session_factory"]() as session:
            resulting_model = session.query(self.data_infrastructure["model"]["model"]).filter(
                getattr(self.data_infrastructure["model"]["model"], "id") == model_id).first()
            resulting_model.path = "my_new_path"
            session.commit()
            session.refresh(resulting_model)
        self.assertEqual(resulting_model.path, "my_new_path")

        instance.model_id = model_id
        with self.data_infrastructure["session_factory"]() as session:
            session.add(instance)
            session.commit()
            session.refresh(instance)
            resulting_model = session.query(self.data_infrastructure["model"]["model"]).filter(
                getattr(self.data_infrastructure["model"]["model"], "id") == model_id).first()
            self.assertEqual(instance.model_id, resulting_model.id)
            self.assertTrue(isinstance(
                instance.model, self.data_infrastructure["model"]["model"]))

    @classmethod
    def setUpClass(cls):
        """
        Class method for setting up test case.
        """
        if not os.path.exists(cfg.PATHS.TEST_PATH):
            os.makedirs(cfg.PATHS.TEST_PATH)
        cls.database = ModelDatabase(
            database_uri=f"sqlite:///{TESTING_DB_PATH}",
            schema="test.",
            verbose=True)

        cls.example_model_data = {
            "path": "TheBloke_vicuna-7B-v1.3-GGML",
            "task": "text-generation",
            "architecture": "llama"
        }
        cls.example_modelversion_data = {
            "path": "vicuna-7b-v1.3.ggmlv3.q4_0.bin",
            "format": "ggml",
            "quantization": "q4_0"
        }
        cls.example_instance_data = {
            "backend": "llamacpp",
            "loader": "_default",
            "loader_kwargs": {
                "n_ctx": 2048,
                "verbose": True
            }
        }
        cls.example_log_data = {"request":
                                {"my_request_key": "my_request_value"}}
        cls.model_columns = ["id", "path", "task", "architecture", "parametercount"
                             "url", "meta_data", "created", "updated", "inactive"]
        cls.modelversion_columns = ["id", "path", "format", "quantization"
                                    "url", "meta_data", "created", "updated", "inactive"]
        cls.instance_columns = ["uuid", "backend", "loader", "loader_kwargs", "gateway",
                                "meta_data", "created", "updated", "inactive", "model_id", "modelversion_id"]
        cls.log_columns = ["id", "request", "response", "started", "finished"]

    @classmethod
    def tearDownClass(cls):
        """
        Class method for setting tearing down test case.
        """
        del cls.data_infrastructure
        del cls.example_model_data
        del cls.example_instance_data
        del cls.example_log_data
        del cls.model_columns
        del cls.instance_columns
        del cls.log_columns
        if os.path.exists(cfg.PATHS.TEST_PATH):
            shutil.rmtree(cfg.PATHS.TEST_PATH, ignore_errors=True)
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
