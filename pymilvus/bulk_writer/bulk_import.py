# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import logging
from typing import Optional

import requests

from pymilvus.exceptions import MilvusException

logger = logging.getLogger("bulk_import")
logger.setLevel(logging.DEBUG)


def _http_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) "
        "Chrome/17.0.963.56 Safari/535.11",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encodin": "gzip,deflate,sdch",
        "Accept-Languag": "en-US,en;q=0.5",
    }


def _throw(msg: str):
    logger.error(msg)
    raise MilvusException(message=msg)


def _post_request(url: str, params: {}, timeout: int = 20, **kwargs):
    try:
        resp = requests.post(
            url=url, headers=_http_headers(), data=params, timeout=timeout, **kwargs
        )
        if resp.status_code < 200 or resp.status_code >= 300:
            _throw(f"Failed to post url: {url}, status code: {resp.status_code}")
        else:
            return resp
    except Exception as err:
        _throw(f"Failed to post url: {url}, error: {err}")


def _get_request(url: str, params: {}, timeout: int = 20, **kwargs):
    try:
        resp = requests.get(
            url=url, headers=_http_headers(), params=params, timeout=timeout, **kwargs
        )
        if resp.status_code < 200 or resp.status_code >= 300:
            _throw(f"Failed to get url: {url}, status code: {resp.status_code}")
        else:
            return resp
    except Exception as err:
        _throw(f"Failed to get url: {url}, error: {err}")


## bulkinsert RESTful api wrapper
def bulk_insert(
    url: str,
    object_url: str,
    access_key: str,
    secret_key: str,
    cluster_id: str,
    collection_name: str,
    partition_name: Optional[str] = None,
    **kwargs,
):
    """call bulkinsert restful interface to import files

    Args:
        url (str): url of the server
        object_url (str): data files url
        access_key (str): access key to access the object storage
        secret_key (str): secret key to access the object storage
        cluster_id (str): id of a milvus instance(for cloud)
        collection_name (str): name of the target collection
        partition_name (str): name of the target partition, 'None' for default partition

    Returns:
        json: response of the restful interface
    """
    params = {
        "objectUrl": object_url,
        "accessKey": access_key,
        "secretKey": secret_key,
        "clusterId": cluster_id,
        "collectionName": collection_name,
        "partitionName": partition_name,
    }

    resp = _post_request(url=url, params=params, **kwargs)
    res = resp.json()
    if "jobID" not in res:
        msg = "Illegal result from bulkinsert call: no job id"
        raise MilvusException(message=msg)
    return res


def get_job_progress(url: str, job_id: str, cluster_id: str, **kwargs):
    """get job progress

    Args:
        url (str): url of the server
        job_id (str): a job id
        cluster_id (str): id of a milvus instance(for cloud)

    Returns:
        json: response of the restful interface
    """
    params = {
        "jobID": job_id,
        "clusterId": cluster_id,
    }

    resp = _get_request(url=url, params=params, **kwargs)
    return resp.json()


def list_jobs(url: str, cluster_id: str, page_size: int, current_page: int, **kwargs):
    """list jobs in a cluster

    Args:
        url (str): url of the server
        cluster_id (str): id of a milvus instance(for cloud)
        page_size (int): pagination size
        current_page (int): pagination

    Returns:
        json: response of the restful interface
    """
    params = {
        "clusterId": cluster_id,
        "pageSize": page_size,
        "currentPage": current_page,
    }

    resp = _get_request(url=url, params=params, **kwargs)
    return resp.json()
