import pickle
import pathlib
import structlog

import pandas
import pandas as pd
from pandas.testing import assert_frame_equal
import os

logger = structlog.getLogger(__name__)

PICKLE_STORE_PATH = pathlib.Path('sandbox/pickle_store')

logger.info('setting cache storage path', path=PICKLE_STORE_PATH)

os.makedirs(PICKLE_STORE_PATH, exist_ok=True)

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

    logger.info('dumping object', file_name=file_name, parse_dates=parse_dates, path=path)

    if type(obj) == pandas.DataFrame:
        # save file
        obj.to_csv(path, compression='gzip', index=False)

        # Save metadata necessary for loading
        non_date_dict = obj[[c for c in obj.columns if c not in parse_dates]].dtypes.to_dict()

        pickle.dump(non_date_dict, open(f"{path}.non_date_dict.pickle", 'wb'))

        pickle.dump(parse_dates, open(f"{path}.parse_dates.pickle", 'wb'))

        pickle.dump(type(obj), open(f'{path}.obj_type.pickle', 'wb'))

        # save index for loading
        pickle.dump(obj.index, open(f'{path}.index.pickle', 'wb'))

        # test that it will actually work as expected
        pd.testing.assert_frame_equal(obj, load_obj_local(file_name=file_name))

    else:
        with open(path, 'wb') as f:
            pickle.dump(obj=obj, file=f)


def load_obj_local(file_name: str) -> object:
    """
    load the object from local ./sandbox/pickle_store/
    Args:
        file_name: the name of the file to store it

    Returns: the object

    """
    path = PICKLE_STORE_PATH / file_name

    object_type = pickle.load(open(f'{path}.obj_type.pickle', 'rb'))

    logger.info('loading object', path=path, object_type=object_type)

    if object_type == pd.DataFrame:
        non_date_dict = pickle.load(open(f"{path}.non_date_dict.pickle", 'rb'))

        parse_dates = pickle.load(open(f"{path}.parse_dates.pickle", 'rb'))

        index = pickle.load(open(f'{path}.index.pickle', 'rb'))

        # Load
        o = pd.read_csv(path, compression='gzip',
                        dtype=non_date_dict,
                        parse_dates=parse_dates)

        for c in o.columns:
            if c.startswith('Unnamed: '):
                o.drop(columns=[c], inplace=True)

        if isinstance(index, pd.Index):
            o.index = index

        return o
    else:
        with open(PICKLE_STORE_PATH / file_name, 'rb') as f:
            return pickle.load(file=f)


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


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


if __name__ == '__main__':
    # example script
    test = pd.DataFrame({'a': [1, 2, 3], 'b': [1., 2., 3.], 'c': ['a', 'b', 'c'],
                         'd': pd.to_datetime(['2023-01-01', '2023-02-02', '2023-03-03'])})

    dump_obj_local(obj=test, file_name='test', parse_dates=['d'])

    test_a = load_obj_local(file_name='test')

    compare_objects(pickle_a='test',
                    pickle_b='test')
