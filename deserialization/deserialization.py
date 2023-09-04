import json
import struct
from typing import List
from entities.input_entity import InputData
from entities.run_entity import Run

def read_binary_file(file_path: str) -> bytes:
    """Read a binary file and return its content as bytes."""
    with open(file_path, 'rb') as file:
        return file.read()


def read_json_file(file_path: str) -> dict:
    """Read a JSON file and return its content as a dictionary."""
    with open(file_path, 'r') as file:
        content = json.load(file)
        if not content:
            raise ValueError("JSON file is empty")
        return content


def extract_key_from_json(json_dict: dict, key: str):
    """Extract a specific key from a JSON dictionary."""
    data = json_dict.get(key)
    if data is None:
        raise ValueError(f"{key} is not found in JSON file")
    return data

def extract_run(json_path: str) -> Run:
    json_content = read_json_file(json_path)
    info = extract_key_from_json(json_content, 'info')
    run = Run(
        player_name= info['playerName'],
        run_start= info['runStart'],
        run_end= info['runEnd'],
        score= info['score'],
        seed= info['seed']
    )
    return run

def deserialize(json_path: str, dat_path: str) -> List[InputData]:
    """Deserialize binary and JSON data into a list of InputData objects."""
    json_content = read_json_file(json_path)
    states_number = extract_key_from_json(json_content, 'statesNumber')
    state_raw_size = extract_key_from_json(json_content, 'stateRawSize')
    info = extract_key_from_json(json_content, 'info')

    player_name = info.get('playerName')
    run_start = info.get('runStart')

    inputs = read_binary_file(dat_path)
    fmt = "q9f3?"

    deserialized_states = []
    offset = 0

    for _ in range(states_number):
        state_data = struct.unpack_from(fmt, inputs, offset)
        input_features = InputData(
            player_name=player_name,
            run_start=run_start,
            # unpack and map state_data to InputData fields
            **dict(zip(list(InputData.__annotations__.keys())[2:], state_data))

        )
        deserialized_states.append(input_features)
        offset += state_raw_size

    return deserialized_states
