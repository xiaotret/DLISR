from core.dataset import dataset
from core.process_raw_data import meta_data

cur_args = None # 参数实例
cur_md = None # meta_data的单例
cur_ds = None # dataset的单例


def get_md():
    global cur_md
    return cur_md


def set_md(args):
    global cur_md
    cur_md = meta_data(args)


def get_ds():
    global cur_ds
    return cur_ds


def set_ds(ds:dataset):
    global cur_ds
    cur_ds = ds


def get_args():
    global cur_args
    return cur_args


def set_args(args):
    global cur_args
    cur_args = args