#!/usr/bin/env python3
"""
为scripts/train.py添加checkpoint保存功能的补丁脚本
"""

import re

# 读取train.py
with open('scripts/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 添加save_interval参数（如果还没有）
if '--save-interval' not in content:
    # 在--eval-interval后添加--save-interval
    content = content.replace(
        '''    parser.add_argument("--eval-interval", type=int,
                        default=yaml_config.get("eval_interval", 500),
                        help="评估间隔")

    # 优化器配置''',
        '''    parser.add_argument("--eval-interval", type=int,
                        default=yaml_config.get("eval_interval", 500),
                        help="评估间隔")
    parser.add_argument("--save-interval", type=int,
                        default=yaml_config.get("save_interval", 100),
                        help="模型保存间隔")

    # 优化器配置'''
    )

# 2. 在创建日志系统后添加checkpoint manager创建（如果还没有）
if 'create_checkpoint_manager' not in content:
    # 在"创建日志系统"后添加checkpoint manager
    content = content.replace(
        '''    console.print(f"✓ 日志系统创建完成")
    console.print(f"  日志目录: {log_dir}")

    # ==================== 创建训练器 ====================''',
        '''    console.print(f"✓ 日志系统创建完成")
    console.print(f"  日志目录: {log_dir}")

    # ==================== 创建检查点管理器 ====================
    console.print("\\n[bold cyan]8.5 创建检查点管理器[/bold cyan]")

    checkpoint_manager = create_checkpoint_manager(
        log_dir=log_dir,
        max_to_keep=5,
        keep_best=True,
        metric_name="mean_reward",
        metric_mode="max",
    )

    console.print(f"✓ 检查点管理器创建完成")
    console.print(f"  检查点目录: {log_dir}/checkpoints")
    console.print(f"  最多保留: 5个检查点")
    console.print(f"  最佳模型指标: mean_reward (越大越好)")

    # ==================== 创建训练器 ===================='''
    )

# 3. 在训练循环的日志输出后添加checkpoint保存（如果还没有）
if 'checkpoint_manager.save_checkpoint' not in content:
    # 在日志输出后添加checkpoint保存
    content = content.replace(
        '''                # 重置指标累积器
                metrics_logger.reset()

        return train_state, env_state, info''',
        '''                # 重置指标累积器
                metrics_logger.reset()

            # 定期保存检查点
            if (update + 1) % args.save_interval == 0:
                avg_metrics_for_save = metrics_logger.get_averages()
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    train_state=train_state,
                    step=train_state.step,
                    metrics=avg_metrics_for_save,
                )
                console.print(f"[dim]💾 检查点已保存: {checkpoint_path}[/dim]")

        return train_state, env_state, info'''
    )

# 4. 在训练完成后保存最终checkpoint（如果还没有）
if '最终检查点' not in content and '训练完成' in content:
    content = content.replace(
        '''        # ==================== 训练完成 ====================
        logger.print_summary(
            "✓ 训练完成！\\n"
            f"总步数: {train_state.step}\\n"
            f"总环境步数: {train_state.env_steps:,}",
            style="green",
        )''',
        '''        # ==================== 训练完成 ====================
        # 保存最终检查点
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            train_state=train_state,
            step=train_state.step,
            metrics=metrics_logger.get_averages(),
            force=True,  # 强制保存
        )
        console.print(f"[green]💾 最终检查点已保存: {final_checkpoint_path}[/green]")

        # 显示最佳模型信息
        best_info = checkpoint_manager.get_best_model_info()
        if best_info:
            console.print(f"[green]🏆 最佳模型信息:[/green]")
            for key, value in best_info.items():
                console.print(f"  {key}: {value}")

        logger.print_summary(
            "✓ 训练完成！\\n"
            f"总步数: {train_state.step}\\n"
            f"总环境步数: {train_state.env_steps:,}",
            style="green",
        )'''
    )

# 写回文件
with open('scripts/train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ train.py已成功更新，添加了checkpoint保存功能")
