from typing import TYPE_CHECKING, Dict

import gradio as gr
from transformers.trainer_utils import SchedulerType

from ...extras.constants import TRAINING_STAGES
from ..common import DEFAULT_DATA_DIR, list_adapters, list_dataset
from ..components.data import create_preview_box
from ..utils import gen_plot


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_train_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        training_stage = gr.Dropdown(
            choices=list(TRAINING_STAGES.keys()), value=list(TRAINING_STAGES.keys())[0], scale=2
        )
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2)
        dataset = gr.Dropdown(multiselect=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    dataset_dir.change(list_dataset, [dataset_dir, training_stage], [dataset], queue=False)

    input_elems.update({training_stage, dataset_dir, dataset})
    elem_dict.update(dict(training_stage=training_stage, dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        learning_rate = gr.Textbox(value="5e-5")
        num_train_epochs = gr.Textbox(value="3.0")
        max_grad_norm = gr.Textbox(value="1.0")
        max_samples = gr.Textbox(value="100000")
        compute_type = gr.Dropdown(choices=["fp16", "bf16", "fp32", "pure_bf16"], value="fp16")

    input_elems.update({learning_rate, num_train_epochs, max_grad_norm, max_samples, compute_type})
    elem_dict.update(
        dict(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_grad_norm=max_grad_norm,
            max_samples=max_samples,
            compute_type=compute_type,
        )
    )

    with gr.Row():
        cutoff_len = gr.Slider(value=1024, minimum=4, maximum=16384, step=1)
        batch_size = gr.Slider(value=2, minimum=1, maximum=1024, step=1)
        gradient_accumulation_steps = gr.Slider(value=8, minimum=1, maximum=1024, step=1)
        val_size = gr.Slider(value=0, minimum=0, maximum=1, step=0.001)
        lr_scheduler_type = gr.Dropdown(choices=[scheduler.value for scheduler in SchedulerType], value="cosine")

    input_elems.update({cutoff_len, batch_size, gradient_accumulation_steps, val_size, lr_scheduler_type})
    elem_dict.update(
        dict(
            cutoff_len=cutoff_len,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            val_size=val_size,
            lr_scheduler_type=lr_scheduler_type,
        )
    )

    with gr.Accordion(label="Extra config", open=False) as extra_tab:
        with gr.Row():
            logging_steps = gr.Slider(value=5, minimum=5, maximum=1000, step=5)
            save_steps = gr.Slider(value=100, minimum=10, maximum=5000, step=10)
            warmup_steps = gr.Slider(value=0, minimum=0, maximum=5000, step=1)
            neftune_alpha = gr.Slider(value=0, minimum=0, maximum=10, step=0.1)
            optim = gr.Textbox(value="adamw_torch")

        with gr.Row():
            resize_vocab = gr.Checkbox()
            sft_packing = gr.Checkbox()
            upcast_layernorm = gr.Checkbox()
            use_llama_pro = gr.Checkbox()
            shift_attn = gr.Checkbox()

    input_elems.update(
        {
            logging_steps,
            save_steps,
            warmup_steps,
            neftune_alpha,
            optim,
            resize_vocab,
            sft_packing,
            upcast_layernorm,
            use_llama_pro,
            shift_attn,
        }
    )
    elem_dict.update(
        dict(
            extra_tab=extra_tab,
            logging_steps=logging_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            neftune_alpha=neftune_alpha,
            optim=optim,
            resize_vocab=resize_vocab,
            sft_packing=sft_packing,
            upcast_layernorm=upcast_layernorm,
            use_llama_pro=use_llama_pro,
            shift_attn=shift_attn,
        )
    )

    with gr.Accordion(label="Freeze config", open=False) as freeze_tab:
        with gr.Row():
            num_layer_trainable = gr.Slider(value=3, minimum=1, maximum=128, step=1, scale=2)
            name_module_trainable = gr.Textbox(value="all", scale=3)

    input_elems.update({num_layer_trainable, name_module_trainable})
    elem_dict.update(
        dict(
            freeze_tab=freeze_tab, num_layer_trainable=num_layer_trainable, name_module_trainable=name_module_trainable
        )
    )

    with gr.Accordion(label="LoRA config", open=False) as lora_tab:
        with gr.Row():
            lora_rank = gr.Slider(value=8, minimum=1, maximum=1024, step=1, scale=1)
            lora_alpha = gr.Slider(value=16, minimum=1, maximum=2048, step=1, scale=1)
            lora_dropout = gr.Slider(value=0.1, minimum=0, maximum=1, step=0.01, scale=1)
            lora_target = gr.Textbox(scale=2)

        with gr.Row():
            use_rslora = gr.Checkbox(scale=1)
            use_dora = gr.Checkbox(scale=1)
            create_new_adapter = gr.Checkbox(scale=1)
            additional_target = gr.Textbox(scale=2)

    input_elems.update(
        {lora_rank, lora_alpha, lora_dropout, lora_target, use_rslora, use_dora, create_new_adapter, additional_target}
    )
    elem_dict.update(
        dict(
            lora_tab=lora_tab,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target=lora_target,
            use_rslora=use_rslora,
            use_dora=use_dora,
            create_new_adapter=create_new_adapter,
            additional_target=additional_target,
        )
    )

    with gr.Accordion(label="RLHF config", open=False) as rlhf_tab:
        with gr.Row():
            dpo_beta = gr.Slider(value=0.1, minimum=0, maximum=1, step=0.01, scale=1)
            dpo_ftx = gr.Slider(value=0, minimum=0, maximum=10, step=0.01, scale=1)
            reward_model = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=2)

    training_stage.change(list_dataset, [dataset_dir, training_stage], [dataset], queue=False).then(
        list_adapters,
        [engine.manager.get_elem_by_name("top.model_name"), engine.manager.get_elem_by_name("top.finetuning_type")],
        [reward_model],
        queue=False,
    )

    input_elems.update({dpo_beta, dpo_ftx, reward_model})
    elem_dict.update(dict(rlhf_tab=rlhf_tab, dpo_beta=dpo_beta, dpo_ftx=dpo_ftx, reward_model=reward_model))

    with gr.Accordion(label="GaLore config", open=False) as galore_tab:
        with gr.Row():
            use_galore = gr.Checkbox(scale=1)
            galore_rank = gr.Slider(value=16, minimum=1, maximum=1024, step=1, scale=2)
            galore_update_interval = gr.Slider(value=200, minimum=1, maximum=1024, step=1, scale=2)
            galore_scale = gr.Slider(value=0.25, minimum=0, maximum=1, step=0.01, scale=2)
            galore_target = gr.Textbox(value="mlp,attn", scale=3)

    input_elems.update({use_galore, galore_rank, galore_update_interval, galore_scale, galore_target})
    elem_dict.update(
        dict(
            galore_tab=galore_tab,
            use_galore=use_galore,
            galore_rank=galore_rank,
            galore_update_interval=galore_update_interval,
            galore_scale=galore_scale,
            galore_target=galore_target,
        )
    )

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button()
        stop_btn = gr.Button()

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                output_dir = gr.Textbox()

            with gr.Row():
                resume_btn = gr.Checkbox(visible=False, interactive=False)
                process_bar = gr.Slider(visible=False, interactive=False)

            with gr.Box():
                output_box = gr.Markdown()

        with gr.Column(scale=1):
            loss_viewer = gr.Plot()

    input_elems.add(output_dir)
    output_elems = [output_box, process_bar]

    cmd_preview_btn.click(engine.runner.preview_train, input_elems, output_elems)
    start_btn.click(engine.runner.run_train, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort, queue=False)
    resume_btn.change(engine.runner.monitor, outputs=output_elems)

    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            output_dir=output_dir,
            resume_btn=resume_btn,
            process_bar=process_bar,
            output_box=output_box,
            loss_viewer=loss_viewer,
        )
    )

    output_box.change(
        gen_plot,
        [
            engine.manager.get_elem_by_name("top.model_name"),
            engine.manager.get_elem_by_name("top.finetuning_type"),
            output_dir,
        ],
        loss_viewer,
        queue=False,
    )

    return elem_dict
