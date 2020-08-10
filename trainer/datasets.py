import torchvision.transforms.functional as f


class WheatDataset(Dataset):
    def __init__(
            self,
            data_path,
            images_folder,
            csv_file,
            train_val_test,
            input_size,
            transform=None,
            classes=None

    ):
        self.transform = transform
        self.data_path = data_path
        self.images_folder = images_folder
        self.classes = classes
        self.train_val_test = train_val_test
        self.input_size = input_size
        if train_val_test == "train" or train_val_test == "val":
            train_path = os.path.join(data_path, csv_file)
            data = pd.read_csv(train_path)
            # several bbox per image, therefore a non duplicates list must be generated
            filtered_data = data.image_id.unique()
            lendata = len(filtered_data)

            num_data_train = int(lendata // 1.25)  # 80% data

            num_test_data = lendata - num_data_train

            lastValueindexes = data.index[data['image_id'] == filtered_data[num_data_train]]
            lastValueindex = lastValueindexes[-1]

            if train_val_test == "train":
                self.num_samples = num_data_train + 1
                self.fnames, self.boxes, self.labels = init_object_detector_dataset(data, 0, lastValueindex, data_path,
                                                                                    images_folder)

            if train_val_test == "val":
                self.num_samples = num_test_data - 1
                end_register = len(data)
                self.fnames, self.boxes, self.labels = init_object_detector_dataset(data, lastValueindex + 1,
                                                                                    end_register, data_path,
                                                                                    images_folder)

        elif train_val_test == "test":
            count = 0
            testpath = os.path.join(self.data_path, "test")
            d = testpath
            for path in os.listdir(d):
                if os.path.isfile(os.path.join(d, path)):
                    count += 1
            self.num_samples = count
            test_folder_path = os.path.join(data_path, "test")
            listfiles = [f for f in listdir(test_folder_path) if os.path.isfile(os.path.join(test_folder_path, f))]

            self.fnames = listfiles
            self.boxes = []
            self.labels = []


    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        prepath = os.path.join(self.data_path, self.images_folder)
        path = os.path.join(prepath, self.fnames[idx])
        img = cv2.imread(path)
        if img is None or np.prod(img.shape) == 0:
            print('cannot load image from path: ', path)
            sys.exit(-1)

        img = img[..., ::-1]
        if self.train_val_test != "test":

            boxes = self.boxes[idx].clone()
            labels = self.labels[idx]
            size = self.input_size

            # Resize & Flip
            img, boxes = resize(img, boxes, (size, size))
            if self.train_val_test == 'train':
                img, boxes = random_flip(img, boxes)
            # Data augmentation.
            img = np.array(img)
            if self.transform:
                img = self.transform(image=img)['image']

            return img, boxes, labels

        size = self.input_size
        #         print(img.shape)
        img = resizeImageOnly(img, (size, size))
        img = np.array(img)
        #         print(img.shape)
        if self.transform:
            img = self.transform(np.squeeze(img, axis=0))
        #             print(img.size)
        return img, self.fnames[idx]

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, w, h)
        encoder = DataEncoder((w, h))
        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = encoder.encode(boxes[i], labels[i])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples

