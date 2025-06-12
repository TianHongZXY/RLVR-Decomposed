# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH dataset to parquet format
"""

import os
import datasets
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.utils import extract_answer_math

def filter_dataset(dataset, question_ids):
    dataset = dataset.select(question_ids)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'TianHongZXY/MATH'

    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    print("Train dataset: ", train_dataset)
    test_dataset = dataset['test']
    print("Test dataset: ", test_dataset)

    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            solution = example.pop('solution')
            answer = extract_answer_math(solution)
            solution = {
                "question": question,
                "solution": solution,
                "target": answer,
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
