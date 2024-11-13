import pandas as pd
from pathlib import Path
import os

def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=["author", "text"])
    df_test = pd.DataFrame(columns=["author", "text"])
    author_to_id_map = {"kennedy": 0, "johnson": 1}

    def make_df_from_dir(dir_name, df):
        """
        Constructing the dataframe from the specific directory
        :param dir_name: name of directory - should be an auther name or unlabeled
        :param df: data frame to encode data to
        :return: a dataframe containing the auther and the text of each presidential speech
        """
        
        for f in path.glob(f"./{dir_name}/*.txt"):
            with open(f, encoding='utf-8') as fp:
                text = fp.read()
                if dir_name in ("kennedy", "johnson"):
                    row = {'author': dir_name, 'text': text}
                    
                elif dir_name == 'unlabeled':
                    row = {'author': f.stem.split('_')[-1], 'text': text}
                    
                df = pd.concat([df,pd.DataFrame([row])], ignore_index=True)
        return df

    for p in path.iterdir():
        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == "unlabeled":
            df_test = make_df_from_dir(p.name, df_test)

    
    # replace the strings for the author names with numeric codes (0, 1)
    df_train["author"] = df_train["author"].apply(lambda x: author_to_id_map.get(x))
    # do the same for the test data
    df_test["author"] = df_test["author"].apply(lambda x: author_to_id_map.get(x))

    
    return df_train, df_test

def build_dataframe_from_docs_txt(filepath):
    df_train = pd.DataFrame(columns=["author", "text"])
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('+') or line.startswith('-'):
                label = line[0]
                text = line[1:].strip()
                row = {'author': label, 'text': text}
                df_train = pd.concat([df_train, pd.DataFrame([row])], ignore_index=True)
    return df_train
