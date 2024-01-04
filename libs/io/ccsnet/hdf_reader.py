import h5py


class h5_thang():
    
    def __init__(
        self, 
        file
    ):
        
        self.file = file
        
    def h5_keys(
        self, 
        verbose=True
    ):

        items = []
        def func(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append(name)

        f = h5py.File(self.file, 'r', locking=False)
        f.visititems(func)

        if verbose:
            for item in items:
                print(item)
        return items
    
    def h5_data(
        self,
        items: list=None,
        verbose: bool=False 
    ):

        if items == None:
            items = self.h5_keys(verbose=Falsef)
            
        if verbose:
            for item in items:
                print(item)
                
        data_dict = {}
        with h5py.File(self.file , 'r', locking=False) as h1:
            for item in items:
                data_dict[item] = h1[item][:]
                    
        return data_dict