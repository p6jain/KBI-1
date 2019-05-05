import torch
import sklearn.decomposition
import sklearn.manifold
import numpy
#from mpl_toolkits.mplot3d import Axes3D
import kb
#import matplotlib.pyplot
import csv
from PIL import Image
from torchvision import transforms

from pathlib import Path
from ast import literal_eval

def plot_weights(w, sne=False):
    w = w.cpu().numpy()
    if sne:
        pca = sklearn.manifold.TSNE(n_components=3)
    else:
        pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(w)
    w = pca.transform(w)
    fig = matplotlib.pyplot.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.scatter(w[:, 0], w[:, 1], w[:, 2])
    matplotlib.pyplot.show()


def plot_weights_2d(w, sne=False, metric='cosine'):
    w = w.cpu().numpy()
    if sne:
        pca = sklearn.manifold.TSNE(n_components=2, verbose=1, metric=metric)
    else:
        pca = sklearn.decomposition.PCA(n_components=2)
    w = pca.fit_transform(w)
    matplotlib.pyplot.scatter(w[:, 0], w[:, 1])
    matplotlib.pyplot.show()


def fb15k_type_map():
    f = open('data/fb15k/entity2type.txt')
    f = [x.split() for x in f.readlines()]
    mp = {}
    def remove_types(fl, tp):
        ls = []
        rest = []
        for x in fl:
            present = False
            for y in x[1:]:
                if y.startswith(tp):
                    present = True
            if not present:
                ls.append(x)
            else:
                rest.append(x)
        return ls, rest
    rest, people = remove_types(f, '/people')
    rest, location = remove_types(rest, '/location')
    rest, organisation = remove_types(rest, '/organisation')
    rest, film = remove_types(rest, '/film')
    rest, sports = remove_types(rest, '/sports')
    def get_set(fl):
        return set([x[0] for x in fl])
    people = get_set(people)
    location = get_set(location)
    organisation = get_set(organisation)
    film = get_set(film)
    sports = get_set(sports)
    rest = get_set(rest)
    types = [people, location, organisation, film, sports, rest]
    def get_type(e):
        for i, t in enumerate(types):
            if e in t:
                return i
        return len(types)-1
    k = kb.kb('data/fb15k/train.txt')
    for i, e in enumerate(k.entity_map):
        mp[i] = get_type(e)
    print(mp)
    return mp


def colored_plot(fb15k_weights, sne=False, td=False, dis='cosine'):
    w = fb15k_weights.cpu().numpy()
    if sne:
        pca = sklearn.manifold.TSNE(n_components=3 if td else 2, metric=dis, verbose=2)
    else:
        pca = sklearn.decomposition.PCA(n_components=3 if td else 2)
    tw = pca.fit_transform(w)
    return tw


def plt(tw, w, td, entity_type_map):
    col = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    fig = matplotlib.pyplot.figure()
    if(td):
        axes = fig.add_subplot(111, projection='3d')
        for i in range(w.shape[0]):
            axes.scatter(tw[i, 0], tw[i, 1], tw[i, 2], c=col[entity_type_map[i]], alpha=0.1)
    else:
        axes = matplotlib.pyplot
        for i in range(w.shape[0]):
            axes.scatter(tw[i, 0], tw[i, 1], c=col[entity_type_map[i]], alpha=0.1)
    matplotlib.pyplot.show()


def type_map_fine(dataset_root):
    fl = open(dataset_root+"/entity_mid_name_type_typeid.txt").readlines()
    fl = [x.strip().split('\t') for x in fl]
    result = {}
    for line in fl:
        result[line[0]] = int(line[3])
    return result

def get_betas(dataset_root, relation_map):
    with open(dataset_root+"/fb15k_rel_beta3.csv") as f:
        reader = csv.reader(f)
        best_beta = dict(reader)
        beta_array = numpy.zeros(len(relation_map), dtype=numpy.float32)
        for k in best_beta.keys():
            if float(best_beta[k]) == 0.0:
                #best_beta[k] = 1e-20
                beta_array[int(relation_map[k])] = -1000#1e-10
            elif float(best_beta[k]) == 1.0:
                beta_array[int(relation_map[k])] = 100#0.999999
            else:
                #best_beta[k] = float(best_beta[k])
                beta_array[int(relation_map[k])] = float(best_beta[k])
    #best_beta = numpy.fromiter(best_beta.values(), dtype=float)
    #return numpy.log(best_beta/(1.0-best_beta))
    #print("Prachi Debug", "beta_array",beta_array)
    tmp = numpy.log(beta_array/(1.0-beta_array))
    #print("Prachi Debug", "beta_array",beta_array[:5], beta_array[-5:] ,tmp[:5],tmp[-5:])
    for k in best_beta.keys():
            if float(best_beta[k]) == 0.0:
                #best_beta[k] = 1e-20
                tmp[int(relation_map[k])] = -1000#1e-10
            elif float(best_beta[k]) == 1.0:
                tmp[int(relation_map[k])] = 1000#0.999999
            
    print("Prachi Debug", "beta_array",beta_array[:5], beta_array[-5:] ,tmp[:5],tmp[-5:])
    return tmp;#numpy.log(beta_array/(1.0-beta_array))


def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image.to(device)

    return image_tensor

def fb15k_entity_image_map():
    path_to_image_folder="data/fb15k/all_resize/"
    ip = open("data/fb15k/mid_image_path.txt").readlines()
    ip = [ele.strip("\n").split("\t") for ele in ip]
    mid_image_map = {}
    for line in ip:
        mid_image_map[line[0]] = load_image(path_to_image_folder+line[1])
    return mid_image_map

def entity_name_map_fine(dataset_root):
    fl = open(dataset_root+"/entity_mid_name_type_typeid.txt").readlines()
    fl = [x.strip().split('\t') for x in fl]
    result = {}
    for line in fl:
        result[line[0]] = line[1]
    return result

def type_name_map_fine(dataset_root):
    fl = open(dataset_root+"/entity_mid_name_type_typeid.txt").readlines()
    fl = [x.strip().split('\t') for x in fl]
    result = {}
    for line in fl:
        result[int(line[3])] = line[2]
    return result


class type_mrr_hook(object):
    def __init__(self, kb, type_id):
        self.type_id = type_id
        self.inverse_rank_total = 0
        self.count = 0
        self.kb = kb
    def __call__(self, s, r, o, ranks, top_predictions, top_scores, expected_type, top_predictions_type):
        f = expected_type.eq(self.type_id).float()
        self.inverse_rank_total += (f/ranks.float()).sum()
        self.count += f.sum()
    def end(self):
        print(self.type_id, self.inverse_rank_total/self.count)
    def start(self):
        self.inverse_rank_total = 0
        self.count = 0

class rwise_mrr_hook(object):
    def __init__(self, kb, relation_count):
        self.relation_count = relation_count
        self.inverse_rank_total = torch.zeros(relation_count).cuda()
        self.count = torch.zeros(relation_count).cuda()
        self.kb = kb
    def __call__(self, s, r, o, ranks, top_predictions, top_scores, expected_type, top_predictions_type):
        self.count.scatter_add_(0, r.squeeze(), torch.FloatTensor([1.0]).cuda().expand_as(r.squeeze()))
        self.inverse_rank_total.scatter_add_(0, r.squeeze(), 1/ranks)
    def end(self):
        result = self.inverse_rank_total/self.count
    def start(self):
        self.inverse_rank_total [:] = 0
        self.count[:] = 0

def load_hooks(hooks, kb):
    result = []
    for hook_param in hooks:
        hook_class = globals()[hook_param['name']]
        hook_param['arguments']['kb'] = kb
        result.append(hook_class(**hook_param['arguments']))
    return result

def check_exists(file_path):
    my_file = Path(file_path)
    return my_file.exists()

def get_entity_relation_id_neg_sensitive(mapping,dataset_root):
    em_path = dataset_root+"/image/entity_map.txt"
    ter_path = dataset_root+"/image/type_entity_range.txt"
    if check_exists(em_path) and check_exists(ter_path):
        f_em=open(em_path).readlines()
        f_ter=open(ter_path).readlines()
        entity_map = {}; reverse_entity_map = {}; type_entity_range = {}
        for ele in f_em:
            ele = ele.strip("\n").split("\t")
            entity_map[ele[0]] = int(ele[1])
            reverse_entity_map[int(ele[1])] = ele[0]
        for ele in f_ter:
            ele = ele.strip("\n").split("\t")
            type_entity_range[int(ele[0])] = set(literal_eval(ele[1]))
    else:
        if mapping is None:
            return None,None
        type_entity_sets={}; entity_map={}; reverse_entity_map={}
        for entity, entity_type in mapping.items():
            if entity_type not in type_entity_sets:
                type_entity_sets[entity_type]=set()
            type_entity_sets[entity_type].add(entity)

        type_entity_range = {}
        count=0
        for typeid, typeset in type_entity_sets.items():
            type_entity_range[typeid] = (count, count+len(typeset)-1)
            for ent in typeset:
                entity_map[ent] = count
                reverse_entity_map[count] = ent
                count+= 1
        #print("DEBUG: ",entity_map, reverse_entity_map, type_entity_range)
        f_em=open(dataset_root+"/image/entity_map.txt","w")
        f_ter=open(dataset_root+"/image/type_entity_range.txt","w")
        for ele in type_entity_range:
            f_ter.write(str(ele)+"\t"+str(list(type_entity_range[ele]))+"\n")
        f_ter.close()
        for ele in entity_map:
            f_em.write(ele+"\t"+str(entity_map[ele])+"\n")
        f_em.close()
    
    return entity_map, reverse_entity_map, type_entity_range
