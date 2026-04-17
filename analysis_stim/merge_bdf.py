import mne,logging
import numpy as np
import os
import mne

def restore_processed_eeg_to_bdf(input_channel,raw_info,processed_eeg_npy, output_bdf_path,txt):
    channel_time_data = processed_eeg_npy.reshape(processed_eeg_npy.shape[0],-1)
 
    # 创建新的 Info 对象基础
    selected_ch_names = [raw_info['ch_names'][i] for i in input_channel]
    # selected_ch_types = [raw_info['chs'][i]['kind'] for i in input_channel]
    all_ch_types = raw_info.get_channel_types()
    selected_ch_types = [all_ch_types[i] for i in input_channel]
    info = mne.create_info(
        ch_names=selected_ch_names,
        sfreq=raw_info['sfreq'],
        ch_types=selected_ch_types
    )
    
    # 复制关键元数据
    info['device_info'] = raw_info.get('device_info', {})
    info['experimenter'] = raw_info.get('experimenter', '')
    # info['proj_name'] = raw_info.get('proj_name', '')
    info['description'] = raw_info.get('description', '')
    
    # 复制每个选定通道的位置信息
    for i, orig_idx in enumerate(input_channel):
        ch_dict = raw_info['chs'][orig_idx].copy()
        # 只保留必要的位置信息字段
        keep_keys = ['loc', 'coord_frame', 'kind', 'ch_name']
        ch_dict = {k: ch_dict[k] for k in keep_keys}
        info['chs'][i].update(ch_dict)

    raw_processed = RawArray(channel_time_data, info)  # 假设数据单位为微伏(μV)，转换为伏特(V)
    if raw_info['meas_date'] is not None:
        raw_processed.set_meas_date(raw_info['meas_date'])
    # 5. 设置电极位置
    montage = raw_info.get_montage()
    if montage:
        raw_processed.set_montage(montage)

    # 6. 导出为EDF '{}//out{}.edf'.format(output_bdf_path,txt)
    # file_name = '{}out{}.edf'.format(output_bdf_path,txt)
    # export_raw(file_name, raw_processed, fmt='edf', overwrite=True)
    # 修改为
    if output_bdf_path.endswith('.bdf'):
        file_name = output_bdf_path  # 直接使用传入的路径
        export_raw(file_name, raw_processed, fmt='bdf', overwrite=True)
    else:
        file_name = '{}out{}.edf'.format(output_bdf_path, txt)
        export_raw(file_name, raw_processed, fmt='edf', overwrite=True)
    print("export success")


def concatenate_bdf_files(file_paths, output_path=None):
    """
    按顺序拼接多个BDF文件，调整Annotations为绝对时间
    
    参数:
        file_paths (list): 按顺序排列的BDF文件路径列表
        output_path (str): 输出拼接后BDF文件的路径
        
    返回:
        combined_raw: 拼接后的Raw对象
    """
    if len(file_paths) < 1:
        raise ValueError("至少需要一个文件进行拼接")
    
    # 初始化变量
    all_raws = []        # 存储所有Raw对象
    all_annotations = []  # 存储所有调整后的Annotations
    cumulative_onset = 0   # 累积时间偏移量（秒）
    
    for i, file_path in enumerate(file_paths):
        # 1. 读取当前BDF文件
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        print(f"处理文件 {i+1}/{len(file_paths)}: {file_path}")
        print(f"  原始持续时间: {raw.times[-1]:.3f}秒, 采样频率: {raw.info['sfreq']:.1f}Hz")
        # print(raw.get_data().shape)
        # 2. 获取原始Annotations
        orig_annot = raw.annotations
        
        # 3. 如果是第一个文件，保留所有原始Annotations
        if i == 0:
            # 存储原始Raw和Annotations
            all_raws.append(raw)
            all_annotations.append(orig_annot)
            
            # 记录第一个文件的总时间作为偏移基准
            cumulative_onset = raw.times[-1]
            logging.info(f"首个文件结束时间: {cumulative_onset:.3f}秒")
            continue
        
        # 4. 后续文件处理
        # 4.1 计算这个文件的持续时间（秒）
        current_duration = raw.times[-1]
        
        # 4.2 如果文件有Annotations，调整onset时间为绝对时间
        modified_annot = None
        if orig_annot is not None and len(orig_annot) > 0:
            # 创建新的onset数组，加上累积偏移量
            new_onset = orig_annot.onset + cumulative_onset
            
            # 创建新的Annotations对象
            modified_annot = mne.Annotations(
                onset=new_onset,
                duration=orig_annot.duration,
                description=orig_annot.description
            )
            print(f"调整 {len(modified_annot)} 个Annotations, 第一个事件时间: {new_onset[0]:.3f}s → {modified_annot.onset[0]:.3f}s")
        else:
            print("没有Annotations需要调整")
            modified_annot = orig_annot
        
        # 4.3 存储调整后的Annotations
        all_annotations.append(modified_annot)
        
        # 4.4 更新当前Raw对象的时间向量
        # 注意: BDF文件的时间信息存储在数据中，不需要直接修改
        # 只需要更新后续的时间偏移量
        # 5. 存储当前Raw对象
        all_raws.append(raw)
        
        # 6. 更新累积时间偏移量（加上当前文件持续时间）
        cumulative_onset += current_duration
        print(f"累积时间偏移量: {cumulative_onset:.3f}秒")
    
    # 7. 将所有Raw文件拼接成一个
    if len(all_raws) > 1:
        combined_raw = mne.concatenate_raws(all_raws, preload=False)
        logging.info(f"拼接完成，总持续时间: {combined_raw.times[-1]:.3f}秒")
    else:
        combined_raw = all_raws[0]

    # 8. 保存文件
    if output_path.lower().endswith('.bdf'):
        try:
            # 尝试保存为 BDF
            mne.export.export_raw(output_path, combined_raw, overwrite=True)
            print(f"拼接后的 BDF 文件已保存到: {output_path}")
        except Exception as e:
            print(f"保存为 BDF 失败: {e}")
            # 回退保存为 FIF 格式
            fif_path = os.path.splitext(output_path)[0] + '.fif'
            print(f"尝试保存为 FIF 格式: {fif_path}")
            combined_raw.save(fif_path, overwrite=True)
            print(f"拼接后的 FIF 文件已保存到: {fif_path}")
    
    return combined_raw


# 使用示例
if __name__ == "__main__":
    # 按顺序排列的BDF文件路径
    bdf_files = [
        r"E:\DENOISE\DeepSeparator-main\analysis_1.19\bdf_01\20260119184411_1.bdf",
        r"E:\DENOISE\DeepSeparator-main\analysis_1.19\bdf_01\20260119184411_2.bdf",
    ]
    
    # 输出文件路径
    output_bdf = r"E:\DENOISE\DeepSeparator-main\analysis_1.19\bdf_01\concatenated_sessions.bdf"
    
    # 拼接文件并调整时间
    combined_raw = concatenate_bdf_files(bdf_files, output_bdf)
    
    # 验证结果
    print("\n验证拼接结果:")
    print(f"总采样点数: {combined_raw.n_times}")
    print(f"总持续时间: {combined_raw.times[-1]:.3f}秒")
    print(f"Annotations数量: {len(combined_raw.annotations)}")
    
    # 如果有Annotations，打印部分信息
    if combined_raw.annotations is not None and len(combined_raw.annotations) > 0:
        print("\n前5个Annotations:")
        for i, (onset, duration, desc) in enumerate(zip(
            combined_raw.annotations.onset,
            combined_raw.annotations.duration,
            combined_raw.annotations.description
        )):
            if i >= 5:
                break
            print(f"  {onset:.3f}s | 持续时间: {duration:.3f}s | 描述: {desc}")