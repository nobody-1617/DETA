import argparse
from paths import PROJECT_ROOT

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
parser.add_argument('--data.train', type=str, default='cu_birds', metavar='TRAINSETS', nargs='+', help="Datasets for training extractors")
parser.add_argument('--data.val', type=str, default='cu_birds', metavar='VALSETS', nargs='+',
                    help="Datasets used for validation")
parser.add_argument('--data.test', type=str, default='cu_birds', metavar='TESTSETS', nargs='+',
                    help="Datasets used for testing")
parser.add_argument('--data.num_workers', type=int, default=32, metavar='NEPOCHS',
                    help="Number of workers that pre-process images in parallel")

# model args (TSA)
default_model_name = 'noname'
parser.add_argument('--model.name', type=str, default="url", metavar='MODELNAME',
                    help="A name you give to the extractor".format(default_model_name))
parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
parser.add_argument('--model.classifier', type=str, default='linear', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")
parser.add_argument('--model.dropout', type=float, default=0, help="Adding dropout inside a basic block of widenet")
parser.add_argument('--model.pretrained', action='store_true', help="Using pretrained model for learning or not")


# test args
parser.add_argument('--test.size', type=int, default=600, metavar='TEST_SIZE',
                    help='The number of test episodes sampled')
parser.add_argument('--test.mode', type=str, choices=['mdl', 'sdl'], default='mdl', metavar='TEST_MODE',
                    help="Test mode: multi-domain learning (mdl) or single-domain learning (sdl) settings")
parser.add_argument('--test.type', type=str, choices=['standard', '1shot', '5shot','10shot'], default='10shot', metavar='LOSS_FN',
                    help="meta-test type, standard varying number of ways and shots as in Meta-Dataset, 1shot for five-way-one-shot and 5shot for varying-way-five-shot evaluation.")
parser.add_argument('--test.distance', type=str, choices=['cos', 'l2'], default='cos', metavar='DISTANCE_FN',
                    help="feature similarity function")
parser.add_argument('--test.loss-opt', type=str, choices=['ncc', 'knn', 'lr', 'svm', 'scm'], default='ncc', metavar='LOSS_FN',
                    help="Loss function for meta-testing, knn or prototype loss (ncc), Support Vector Machine (svm), Logistic Regression (lr) or Mahalanobis Distance (scm)")
parser.add_argument('--test.feature-norm', type=str, choices=['l2', 'none'], default='none', metavar='LOSS_FN',
                    help="normalization options")

# path args
parser.add_argument('--model.dir', default='', type=str, metavar='PATH',
                    help='path of single domain learning models')
parser.add_argument('--out.dir', default='', type=str, metavar='PATH',
                    help='directory to output the result and checkpoints')
parser.add_argument('--source', default='', type=str, metavar='PATH',
                    help='path of pretrained model')


parser.add_argument('--maxIt', type=int, default=40, metavar='MaxIteration',
                    help='the value of Max Iteration')
parser.add_argument('--n_regions', type=int, default=2, metavar='N_Regions',
                    help='number of regions cropped from every image')
parser.add_argument('--ratio', type=float, default=0., metavar='Ratio',
                    help='ratio of support samples with data bias')
parser.add_argument('--ours', action='store_true', help="ours or baseline")
parser.add_argument('--pretrained_model', type=str, default="MOCO", metavar='MODELNAME',
                    help="a model to be adapted")

# log args
args = vars(parser.parse_args())
if not args['model.dir']:
    args['model.dir'] = PROJECT_ROOT
if not args['out.dir']:
    args['out.dir'] = args['model.dir']

BATCHSIZES = {
                "ilsvrc_2012": 448,
                "omniglot": 64,
                "aircraft": 64,
                "cu_birds": 64,
                "dtd": 64,
                "quickdraw": 64,
                "fungi": 64,
                "vgg_flower": 64
                }

LOSSWEIGHTS = {
                "ilsvrc_2012": 1,
                "omniglot": 1,
                "aircraft": 1,
                "cu_birds": 1,
                "dtd": 1,
                "quickdraw": 1,
                "fungi": 1,
                "vgg_flower": 1
                }

# lambda^f in our paper
KDFLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1
                }
# lambda^p in our paper
KDPLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1
                }
# k in our paper
KDANNEALING = {
                    "ilsvrc_2012": 5,
                    "omniglot": 2,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 2,
                    "fungi": 2,
                    "vgg_flower": 1
                }
