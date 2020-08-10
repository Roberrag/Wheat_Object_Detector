class Experiment:
    def __init__(
            self,
            system_config: SystemConfig = SystemConfig(),
            dataset_config: DatasetConfig = DatasetConfig(),  # pylint: disable=redefined-outer-name
            dataloader_config: DataloaderConfig = DataloaderConfig(),  # pylint: disable=redefined-outer-name
            optimizer_config: OptimizerConfig = OptimizerConfig(),  # pylint: disable=redefined-outer-name
    ):
        data_root = ""
        imagesFolder = "train"
        testFolder = "test"
        csv_file = "train.csv"
        self.system_config = system_config
        setup_system(system_config)
        self.dataset_train = WheatDataset(
            data_root,
            imagesFolder,
            csv_file,
            "train",
            input_size=300,
            transform=Compose(dataset_config.train_transforms),
            classes=["__background__", "wheat"]
        )


        self.loader_train = DataLoader(
            dataset=self.dataset_train,
            batch_size=dataloader_config.batch_size,
            shuffle=True,
            collate_fn=self.dataset_train.collate_fn,
            num_workers=dataloader_config.num_workers,
            pin_memory=True
        )

        self.dataset_test = WheatDataset(
            data_root,
            imagesFolder,
            csv_file,
            "val",
            input_size=300,
            transform=Compose([Normalize(), ToTensorV2()]),
            classes=["__background__", "wheat"]
        )



        self.loader_test = DataLoader(
            dataset=self.dataset_test,
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            collate_fn=self.dataset_test.collate_fn,
            num_workers=dataloader_config.num_workers,
            pin_memory=True
        )
        self.model = Detector(len(self.dataset_train.classes))


        self.loss_fn = DetectionLoss(len(self.dataset_train.classes))

        self.metric_fn = APEstimator(classes=self.dataset_test.classes)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay

        )
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )

        self.visualizer = MatplotlibVisualizer()

    def run(self, trainer_config: TrainerConfig):
        setup_system(self.system_config)
        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter("image"),
            target_getter=itemgetter("target"),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("mAP"),
            visualizer=self.visualizer,
            model_save_best=trainer_config.model_save_best,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        model_trainer.register_hook("train", train_hook_detection)
        model_trainer.register_hook("test", test_hook_detection)
        model_trainer.register_hook("end_epoch", end_epoch_hook_detection)
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics

    def draw_bboxes(self, rows, columns, trainer_config: TrainerConfig):
        # load the best model
        if trainer_config.model_save_best:
            self.model.load_state_dict(
                torch.
                    load(os.path.join(trainer_config.model_dir, self.model.__class__.__name__) + '_best.pth')
            )
        # or use the last saved
        self.model = self.model.eval()
        self.model = self.model.to("cuda")

        std = (0.229, 0.224, 0.225)
        mean = (0.485, 0.456, 0.406)

        std = torch.Tensor(std)
        mean = torch.Tensor(mean)

        fig, ax = plt.subplots(
            nrows=rows, ncols=columns, figsize=(10, 10), gridspec_kw={
                'wspace': 0,
                'hspace': 0.05
            }
        )

        for axi in ax.flat:
            index = random.randrange(len(self.loader_test.dataset))

            image, gt_boxes, _ = self.loader_test.dataset[index]


            device = torch.device(trainer_config.device)
            image = image.to(device).clone()


            loc_preds, cls_preds = self.model(image.unsqueeze(0))
            with torch.no_grad():
                img = image.cpu()
                img.mul_(std[:, None, None]).add_(mean[:, None, None])
                img = torch.clamp(img, min=0.0, max=1.0)
                img = img.numpy().transpose(1, 2, 0)

                img = (img * 255.).astype(np.uint8)
                gt_img = img.copy()
                pred_img = img.copy()

                for box in gt_boxes:
                    gt_img = cv2.rectangle(
                        gt_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0),
                        thickness=2
                    )
                print(img.shape)
                encoder = DataEncoder((img.shape[1], img.shape[0]))
                samples = encoder.decode(loc_preds, cls_preds)
                c_dets = samples[0][1]  # detections for class == 1
                #                 print(c_dets)

                if c_dets.size > 0:
                    boxes = c_dets[:, :4]
                    for box in boxes:
                        #                         print("xmin: {}, ymin: {}, xmax: {}, ymax: {}".format(int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                        pred_img = cv2.rectangle(
                            pred_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                            thickness=2
                        )

                merged_img = np.concatenate((gt_img, pred_img), axis=1)
                axi.imshow(merged_img)
                axi.axis('off')
        fig.show()