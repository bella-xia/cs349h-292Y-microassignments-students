from torchvision.datasets import MNIST
from torch.utils.data import Subset
import tqdm
import torch.utils
from hdc import *
import itertools
import PIL

MAX = 26


class MNISTClassifier:

    def __init__(self, ber=0.0):
        self.classifier = HDItemMem()

        # initialize all codebooks
        self.pix_cb = HDCodebook('pixel')
        self.coord_cb = HDCodebook('coordinate')
        self.clas_mem = HDItemMem('class', ber=ber)

        # generate pixel color hypervector
        self.pix_cb.add(0)
        self.pix_cb.add(1)
        
        # generate location hypervector
        self.coord_cb.add('row')
        self.coord_cb.add('col')

        return

        # unreachable
        raise Exception("initialize other stuff here")

    def encode_coord(self, i, j):
        row_encode : np.ndarray = HDC.permute(self.coord_cb.get('row'), i)
        col_encode : np.ndarray = HDC.permute(self.coord_cb.get('col'), j)
        return HDC.bind(row_encode, col_encode)

        # unreachable
        raise Exception("encode a coordinate in the image as a hypervector")

    def encode_pixel(self, i, j, v):
        coord_encode : np.ndarray = self.encode_coord(i, j)
        pix_encode : np.ndarray = self.pix_cb.get(v // 255)
        return HDC.bind(coord_encode, pix_encode)

        # unreachable
        raise Exception("encode a pixel in the image as a hypervector")

    def encode_image(self, image):
        encoded_pixs : list[np.ndarray] = []
        for i, j in itertools.product(range(MAX), range(MAX)):
            v = image.getpixel((i, j))
            encoded_pixs.append(self.encode_pixel(i, j, v))
        
        return HDC.bundle(encoded_pixs)        
            
        # unreachable
        raise Exception("return hypervector encoding of image")

    def decode_pixel(self, image_hypervec, i, j):
        coord_encode : np.ndarray = self.encode_coord(i, j)
        pix_encode : np.ndarray = HDC.bind(image_hypervec, coord_encode)
        approx_pix, _ = self.pix_cb.wta(pix_encode)
        return approx_pix

        # unreachable
        raise Exception("retrieve the value of the pixel at coordinate i,j in the image hypervector")

    def decode_image(self, image_hypervec):
        im = PIL.Image.new(mode="1", size=(MAX, MAX))
        for i, j in list(itertools.product(range(MAX), range(MAX))):
            v = self.decode_pixel(image_hypervec, i, j)
            im.putpixel((i, j), v)
        return im

    def train(self, train_data):
        clas_map = {}
        for image, label in tqdm.tqdm(list(train_data)):
            img_encode : np.ndarray = self.encode_image(image)
            if clas_map.get(label, None):
                clas_map[label].append(img_encode)
            else:
                clas_map[label] = [img_encode]
            
        for lab, img_encodes in clas_map.items():
            self.clas_mem.add(lab, HDC.bundle(img_encodes))
        
        return

        # unreachable
        raise Exception("do something with the image, label pair from the dataset")

    def classify(self, image):
        img_encode : np.ndarray = self.encode_image(image)
        return self.clas_mem.wta(img_encode)

        # unreachable
        raise Exception("classify an image using your classifier and return the label and distance")
        return label, dist

    def build_gen_model(self, train_data):
        self.gen_model = {}
        gen_buf = {}
        for image, label in tqdm.tqdm(list(train_data)):
            img_encode : np.nadday = self.encode_image(image)
            if gen_buf.get(label, None):
                gen_buf[label].append(img_encode)
            else:
                gen_buf[label] = [img_encode]
        
        for label, buf in gen_buf.items():
            self.gen_model[label] = np.mean(buf, axis=0)
        return

        # unreachable
        raise Exception("build generative model")

    def generate(self, cat, trials=10):
        gen_hv : np.ndarray = self.gen_model[cat]
        rand_hv : np.ndarray = (np.random.rand(*gen_hv.shape) < gen_hv).astype(np.uint8)
        rand_img = self.decode_image(rand_hv)
        return rand_img

        # unreachable
        raise Exception("generate random image with label <cat> using generative model. Average over <trials> trials.")


def initialize(N=1000):
    alldata = MNIST(root='data', train=True, download=True)
    dataset = list(map(lambda datum: (datum[0].convert("1"), datum[1]),  \
                Subset(alldata, range(N))))

    train_data, test_data = torch.utils.data.random_split(dataset, [0.6, 0.4])
    HDC.SIZE = 10000
    classifier = MNISTClassifier()
    return train_data, test_data, classifier


def test_encoding():
    train_data, test_data, classifier = initialize()
    image0, _ = train_data[0]
    hv_image0 = classifier.encode_image(image0)
    result = classifier.decode_image(hv_image0)
    image0.save("sample0.png")
    result.save("sample0_rec.png")


def test_classifier():
    train_data, test_data, classifier = initialize(2000)

    print("======= training classifier =====")
    classifier.train(train_data)

    print("======= testing classifier =====")
    correct, count = 0, 0
    for image, category in (pbar := tqdm.tqdm(test_data)):
        cat, dist = classifier.classify(image)
        if cat == category:
            correct += 1
        count += 1

        pbar.set_description("accuracy=%f" % (float(correct)/count))

    print("ACCURACY: %f" % (float(correct)/count))

def test_generative_model():
    train_data, test_data, classifier = initialize(1000)
    print("======= building generative model =====")
    classifier.build_gen_model(train_data)

    print("======= generate images =====")
    while True:
        cat = random.randint(0, 9)
        img = classifier.generate(cat)
        print("generated image for class %d" % cat)
        img.save("generated.png")
        input("press any key to generate new image..")


if __name__ == '__main__':
    test_encoding()
    test_classifier()
    test_generative_model()
