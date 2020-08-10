if __name__ == '__main__':
    batch_size_to_set = 30
    epoch_num = 200
    lr_used = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    lr_step_milestones = [40, 100]
    lr_gamma = 0.1
    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=epoch_num, batch_size_to_set=batch_size_to_set)

    #     torch.autograd.set_detect_anomaly(True)
    dataset_config = DatasetConfig(
        root_dir="train",
        train_transforms=[
            RandomBrightness(p=0.5),
            RandomContrast(p=0.5),
            OneOf([
                RandomGamma(),
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50),
                RGBShift()
            ],
                p=1),
            OneOf([Blur(always_apply=True), GaussNoise(always_apply=True)], p=1),
            #             OneOf([HorizontalFlip(p=0.5),
            #                 ShiftScaleRotate(p=0.5)], p=1),
            OneOf([MultiplicativeNoise(multiplier=0.5, p=1),
                   ]),
            CLAHE(),
            Normalize(),
            ToTensorV2()
        ]
    )

    optimizer_config = OptimizerConfig(
        learning_rate=lr_used,
        lr_step_milestones=lr_step_milestones,
        lr_gamma=lr_gamma,
        momentum=momentum,
        weight_decay=weight_decay
    )

    experiment = Experiment(
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        optimizer_config=optimizer_config
    )

    # Run the experiment / start training
    experiment.run(trainer_config)

    # how good our detector works by visualizing the results on the randomly chosen test images:
    experiment.draw_bboxes(4, 1, trainer_config)