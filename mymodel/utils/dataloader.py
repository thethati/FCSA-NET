import torch
import torch.utils.data
from torch_geometric.data import Data


class Batch(Data):
    r""" merge many of small graph to a big graph
    """

    def __init__(self, **kwargs):
        super(Batch, self).__init__(**kwargs)

    @staticmethod
    def from_data_list(data_list):

        # data_list: [[],[],[]]
        # []: (pos, neg, neg, ..., neg) Note: one pos, many neg

        r""" re-assign edge index, due to the edge index must be unique in one graph.
        Concretely, add offset to each edge index of the small graph, and to construct a big graph.
        """

        keys = data_list[0][0].keys
        #print(keys) # ['x', 'edge_index', 'rcid_index', 'y', 'file_items', 'type_embedding', 'scid_index', 'edge_weight']

        assert 'slices_indicator' not in keys

        # slices_indicator is number of nodes in each graph
        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['slices_indicator']:
            batch[key] = []
        
        # print(batch)
        #Batch(edge_index=[0], type_embedding=[0], scid_index=[0], file_items=[0], x=[0], edge_weight=[0], rcid_index=[0], y=[0], slices_indicator=[0])
        #Batch(scid_index=[0], file_items=[0], rcid_index=[0], type_embedding=[0], txt=[0], edge_weight=[0], y=[0], edge_index=[0], x=[0], slices_indicator=[0])

        # keep list structure, only use in processing of the raw dataset
        delattr(batch, "file_items")
        if hasattr(batch, "outfit_cates"):
            delattr(batch, "outfit_cates")

        #print(batch)
        # Batch(y=[0], scid_index=[0], edge_weight=[0], edge_index=[0], x=[0], rcid_index=[0], type_embedding=[0], slices_indicator=[0])

        # record the offset of the edge index
        shift = 0
        for pair_data in data_list:
            for data in pair_data:
                for key in batch.keys:
                    if key == 'slices_indicator':
                        batch[key].append(torch.tensor([shift], dtype=torch.int))
                    elif key == 'edge_index':
                        batch[key].append(data[key] + shift)
                    else:
                        batch[key].append(data[key])
                shift += data.num_nodes
        batch['slices_indicator'].append(torch.tensor([shift], dtype=torch.int))

        tmp_data = data_list[0][0]
        try:
            for key in batch.keys:
                if key == 'slices_indicator':
                   continue             
                else:
                    cat_dim = tmp_data.__cat_dim__(key, tmp_data[key])
                    batch[key] = torch.cat(batch[key], dim=cat_dim)

        except BaseException:
                print(key)
                print(tmp_data[key])
                exit()
       
        # if you need edge_weight
        #batch['edge_weight'].unsqueeze_(-1)

     
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        if hasattr(self, 'slices_indicator'):
            return len(self.__getitem__('slices_indicator')) - 1
        return None


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=Batch.from_data_list, **kwargs)



        
