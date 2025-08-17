def apply_smoothquant_and_find_quantized_accuracy(model: torch.nn.Module, 
                                              evaluator: aimet_common.defs.EvalFunction,
                                              data_loader: torch_data.DataLoader, 
                                              use_cuda: bool = False,
                                              logdir: str = '') -> float:
    """
    Apply SmoothQuant optimization and quantization to the model.
    Similar to adaround workflow to maintain compatibility with AIMET/Qualcomm format.

    :param model: The loaded model
    :param evaluator: The evaluation function
    :param data_loader: DataLoader for calibration
    :param use_cuda: Whether to use CUDA
    :param logdir: Directory for saving encodings
    :return: Accuracy of quantized model
    """
    # First apply batch norm folding like in adaround
    bn_folded_model = copy.deepcopy(model)
    _ = fold_all_batch_norms(bn_folded_model, input_shapes=(1, 3, 224, 224))

    # Create dummy input
    input_shape = (1, 3, 224, 224)  # Adjust as needed
    dummy_input = torch.rand(input_shape)
    if use_cuda:
        dummy_input = dummy_input.cuda()

    # Apply SmoothQuant optimization
    params = SmoothQuantParameters(
        data_loader=data_loader,
        num_batches=5,
        alpha=0.5,
        percentile=99.9,
        channel_wise=True,
        use_empirical_scaling=True
    )
    
    # Apply SmoothQuant optimization
    smoothed_model = SmoothQuant.apply_smooth_quant(
        model=bn_folded_model,
        dummy_input=dummy_input,
        params=params,
        path=logdir,
        filename_prefix='smoothquant'
    )

    # Create quantization simulation model
    quantsim = QuantizationSimModel(
        model=smoothed_model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        rounding_mode='nearest',
        default_output_bw=8,
        default_param_bw=8,
        in_place=False
    )

    # Load and freeze the parameter encodings from SmoothQuant
    quantsim.set_and_freeze_param_encodings(
        encoding_path=os.path.join(logdir, 'smoothquant.encodings')
    )

    # Compute activation encodings
    quantsim.compute_encodings(
        forward_pass_callback=partial(evaluator, use_cuda=use_cuda),
        forward_pass_callback_args=5
    )

    # Export the quantized model
    quantsim.export(
        path=logdir,
        filename_prefix='smoothquant_quantized',
        dummy_input=dummy_input.cpu()
    )

    # Evaluate accuracy
    accuracy = evaluator(quantsim.model, use_cuda=use_cuda)
    return accuracy
