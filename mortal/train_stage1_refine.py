from train_supervised import train


def main():
    train(
        config_section='stage1',
        stage_label='Oracle Dropout Supervised Refinement (Stage 1)',
        checkpoint_label='stage1',
        export_normal_checkpoints=True,
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
