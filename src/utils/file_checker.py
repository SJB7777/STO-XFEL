import h5py

from cuptlib_config.palxfel import load_palxfel_config

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')

if __name__ == "__main__":
    from rocking.rocking_scan import ReadRockingH5

    config = load_palxfel_config("config.ini")
    file29 = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=156\\scan=001\\p0029.h5"
    # with h5py.File(file29) as hf:
    #     print(hf)
    #     h5_tree(hf)

    file30 = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=156\\scan=001\\p0030.h5"
    # with h5py.File(file30) as hf:
    #     print(hf)
    #     h5_tree(hf)

    file53 = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=156\\scan=001\\p0053.h5"
    # with h5py.File(file53) as hf:
    #     print(hf)
    #     h5_tree(hf)
        
    
    rr = ReadRockingH5(file30)