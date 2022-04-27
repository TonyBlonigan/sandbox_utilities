import pickle
import pathlib
import warnings

import pandas
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

PICKLE_STORE_PATH = pathlib.Path('sandbox/pickle_store')


# notes for future object type checking
# df.equals(other_df)
# series.equals(other_series)
# np.allclose(A1, A2)etc


def dump_obj_local(obj: object, file_name: str, parse_dates: list = []) -> None:
    """
    store the object in the local ./sandbox/pickle_store/
    Args:
        obj: the object to be pickled
        file_name: the name of the file to store it
        parse_dates: columns to parse as dates on read

    Returns: None

    """
    assert isinstance(parse_dates, list)
    assert isinstance(file_name, str)
    path = PICKLE_STORE_PATH / file_name
    if type(obj) == pandas.DataFrame:
        # save file
        # obj.to_pickle(path=PICKLE_STORE_PATH / pickle_name)
        obj.to_csv(path, compression='gzip')

        non_date_dict = obj[[c for c in obj.columns if c not in parse_dates]].dtypes.to_dict()
        pickle.dump(non_date_dict, open(f"{path}.non_date_dict.pickle", 'wb'))
        # np.save(f"{path}.non_date_dict.npy", non_date_dict, allow_pickle=True)

        # Save date dtypes
        pickle.dump(parse_dates, open(f"{path}.parse_dates.pickle", 'wb'))

        # save index
        pickle.dump(obj.index.name, open(f'{path}.index.pickle', 'wb'))

        # test that it will actually work as expected
        load_obj_local_dtypes = load_obj_local(file_name=file_name).dtypes
        try:
            assert all(obj.dtypes == load_obj_local_dtypes)
        except AssertionError as e:
            warnings.warn(
                f'dtype mismatch on reload\nobj.dtypes {obj.dtypes}\n\nload_obj_local.dtypes {load_obj_local_dtypes}')
            raise e
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj=obj, file=f)


def load_obj_local(file_name: str, dataframe: bool = True) -> object:
    """
    load the object from local ./sandbox/pickle_store/
    Args:
        file_name: the name of the file to store it
        dataframe: whether pandas data frame

    Returns: the object

    """
    path = PICKLE_STORE_PATH / file_name
    if dataframe:
        non_date_dict = pickle.load(open(f"{path}.non_date_dict.pickle", 'rb'))
        parse_dates = pickle.load(open(f"{path}.parse_dates.pickle", 'rb'))
        index = pickle.load(open(f'{path}.index.pickle', 'rb'))

        # Load
        o = pd.read_csv(path, compression='gzip',
                        dtype=non_date_dict,
                        parse_dates=parse_dates,
                        index_col=index)

        if not index:
            index = [c for c in o.columns if c.startswith('Unnamed: ')]

            o.set_index(keys=index, inplace=True)
    else:
        with open(PICKLE_STORE_PATH / file_name, 'rb') as f:
            o = pickle.load(file=f)

    return o


def compare_objects(pickle_a: str, pickle_b: str) -> None:
    """
    Highlight differences between pickle objects a and b.

    Args:
        pickle_a: pickle file a name
        pickle_b: pickle file b name

    Returns: comparison summary

    """
    print(f'comparing {pickle_a} (a) to {pickle_b} (b)')

    # load the objects
    with open(PICKLE_STORE_PATH / pickle_a, 'rb') as f:
        obj_a = pickle.load(f)

    with open(PICKLE_STORE_PATH / pickle_b, 'rb') as f:
        obj_b = pickle.load(f)

    # compare the objects
    try:
        assert type(obj_a) == type(obj_b)
        print('objects a and b are the same type')
    except AssertionError as e:
        print('objects a and be are different types:', type(obj_a), type(obj_a))

    if isinstance(obj_a, pd.DataFrame):
        assert_frame_equal(obj_a, obj_b, check_like=True)
        print('data frames a and b are equalß')


if __name__ == '__main__':
    compare_objects(pickle_a='load_feats_df.pickle',
                    pickle_b='load_feats_new_df.pickle')
