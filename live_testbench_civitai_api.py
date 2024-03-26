# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import requests
import json
import sqlalchemy
from typing import List
from src.utility.bronze import json_utility
from src.utility.gold.filter_mask import FilterMask
from src.configuration import configuration as cfg
from src.model.model_control.model_database import ModelDatabase, DEFAULT_DB_PATH

from src.model.model_control.api_wrapper import CivitaiAPIWrapper





if __name__ == "__main__":
    """if os.path.exists(DEFAULT_DB_PATH):
       os.remove(DEFAULT_DB_PATH)"""
    db = ModelDatabase(database_uri=None, schema="bronze", verbose=True)

    def callback(model_entries: List[dict]) -> None:
        """
        Callback for model entries.
        :param model_entries: Model entries.
        """
        for entry in model_entries:
            try:
                db.put_object("model_entry", ["url"], **{
                    "url": f"https://civitai.com/api/v1/models/{entry['id']}",
                    "source": "civitai.com",
                    "meta_data": entry
                })
            except sqlalchemy.exc.IntegrityError: 
                pass
    wrapper = CivitaiAPIWrapper()
    wrapper.scrape_available_targets("model", callback=callback, start_url="https://civitai.com/api/v1/models?limit=100&page=190")
