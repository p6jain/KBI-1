import torchvision
import torchvision.models as torchvision_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from ast import literal_eval
from torchvision import transforms
from PIL import Image
import numpy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = torchvision_models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        #self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images, flag_debug=0):
        """Extract feature vectors from input images."""
        images = images.float()
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        #features = self.bn(self.linear(features))
        return features

def augment_image_information(entity_map, entity_mid_image_map):
    """
    Augments the current knowledge base with entity type information for more detailed evaluation.\n
    :param mapping: The maping from entity to types. Expected to be a int to int dict
    :return: None
    """

    entity_id_image_map = {}
    for x in entity_mid_image_map:
        entity_id_image_map[entity_map[x]] = entity_mid_image_map[x]
    #size_details = tuple([len(self.entity_mid_image_map)]+list(self.entity_mid_image_map[x].shape[1:]))
    size_details = tuple([len(entity_map)]+list(entity_mid_image_map[x].shape[1:]))
    entity_id_image_matrix = numpy.zeros(size_details)
    oov_image=numpy.random.rand(1, 3, 224, 224);oov_count=0
    for x in entity_map:#self.entity_mid_image_map:
        if x in entity_mid_image_map.keys():
            entity_id_image_matrix[entity_map[x]] = entity_mid_image_map[x]
        else:
            entity_id_image_matrix[entity_map[x]] = oov_image
            oov_count+=1
    print("OOV images: %d" %oov_count)
    entity_id_image_matrix_np = numpy.array(entity_id_image_matrix, dtype = numpy.long)#
    #entity_id_image_matrix = torch.from_numpy(numpy.array(entity_id_image_matrix))
    return entity_id_image_matrix

'''
Real data access
'''
def fb15k_entity_image_map():
    path_to_image_folder="data/fb15k/all_resize/"
    ip = open("data/fb15k/mid_image_path.txt").readlines()
    ip = [ele.strip("\n").split("\t") for ele in ip]
    mid_image_map = {}
    for line in ip:
        mid_image_map[line[0]] = load_image(path_to_image_folder+line[1])
    return mid_image_map

def get_entity_relation_id_neg_sensitive():
    em_path = "data/fb15k/image/entity_map.txt"
    ter_path = "data/fb15k/image/type_entity_range.txt"
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
    return entity_map, reverse_entity_map, type_entity_range


'''
AUX
'''
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


if __name__ == "__main__":
    entity_map, _, _= get_entity_relation_id_neg_sensitive()
    entity_id_image_map = fb15k_entity_image_map()
    entity_id_image_matrix = augment_image_information(entity_map, entity_id_image_map)

    embed_size = 19
    entity_encoded_image_matrix = numpy.zeros((entity_id_image_matrix.shape[0],2048))
    batch_size = 100

    encoder = EncoderCNN(embed_size).to(device)
    encoder.eval()#to disable dropout and batch normalization
    for i in tqdm(range(0,entity_id_image_matrix.shape[0],batch_size)):
        # Set mini-batch dataset
        image = entity_id_image_matrix[i:i+batch_size]
        image = torch.from_numpy(image)
        image = image.to('cuda')
        # Forward
        feature = encoder(image)
        entity_encoded_image_matrix[i:i+batch_size] = feature

    entity_encoded_image_matrix.dump("data/fb15k/image/image_embeddings_resnet152.dat")
