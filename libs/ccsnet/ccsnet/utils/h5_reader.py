import h5py

class h5_thang():
    
    """
    This function severs for loading all groups/name of h5 compressed data to a dictionary.
    The data can be eazily access by the provided groups/name. 
    """
    
    def __init__(
        self, 
        file
    ):

        self.file = file


    def h5_attrs(
        self,
    )->dict:
        

        attrs_dict = {}

        with h5py.File(self.file , "r", locking=False) as h1:
            
            keys = list(h1.keys()) 

            for key in keys:
                
                for attr in h1[key].attrs.keys():
                    
                    attrs_dict[f"{key}/{attr}"] = h1[key].attrs[attr]
        
        return attrs_dict
    
    
    def h5_keys(
        self, 
        verbose=False
    )->list: 
        
        """
        Args:
            verbose (bool, optional): Defaults to False. If true print out all groups/name.
        Returns:
            list: A list of groups/name.
        """
        
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
        verbose: bool=False, 
        n_data=None
    )->dict:

        """
        Args:
            items (list): A list of groups/name to require from the compressed h5 file.
        Returns:
            dict: A dictionary that contains all direct accessible data
        """
        if items == None:
            items = self.h5_keys(verbose=False)
            
        if verbose:
            for item in items:
                print(item)
                
        data_dict = {}
        with h5py.File(self.file , 'r', locking=False) as h1:
            for item in items:
                if n_data is not None:
                    data_dict[item] = h1[item][:n_data, ...]
                    
                data_dict[item] = h1[item][:]
                    
        return data_dict
