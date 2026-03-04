import os
import shutil
import random
import xlwt,xlrd
def count_images_in_folder(folder_path):
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名，你可以根据实际情况添加更多扩展名
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_count += 1
    return image_count
# 将图片复制到新文件夹中
def copy_images(source_case_folder, destination_case_folder):
    os.makedirs(destination_case_folder, exist_ok=True)
    for image_file in os.listdir(source_case_folder):
        source_path = os.path.join(source_case_folder, image_file)
        destination_path = os.path.join(destination_case_folder, image_file)
        shutil.copy(source_path, destination_path)
if __name__ == '__main__':
    # 设置随机数种子
    random_seed = 1000  # 可以使用任何整数作为种子 #EC原始分布 随机种子42，比例0.6
    #50, 100, 200, 500, 1000
    random.seed(random_seed)
    # 设置训练集和测试集的比例0.5,0.55,0.65,0.70.8
    train_ratio = 0.8  # 80%的图片用于训练，20%用于测试

    data_source="./data/Endometrial_Cancer"
    CT_type="增强"
    workbook=xlwt.Workbook()
    for center in os.listdir(data_source):
        if center.split(".")[-1] in ["xlsx","xls",".DS_Store"]:
            continue
        #设置sheet
        worksheet=workbook.add_sheet(center)
        worksheet.write(0,0,"patient_name")
        worksheet.write(0,1,"dataset")
        worksheet.write(0,2,"label")
        # 设置原始图片文件夹路径
        source_folder = os.path.join(data_source,center,CT_type)
        # 设置新文件夹路径
        name=center.split("_")[-2]
        new_folder = "%s_%s_%d_%f/%s"%(data_source,CT_type,random_seed,train_ratio,name)
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        train_data_folder = new_folder + "/traindata"
        test_data_folder = new_folder + "/testdata"

        # 创建新文件夹结构
        os.makedirs(train_data_folder + "/0", exist_ok=True)
        os.makedirs(train_data_folder + "/1", exist_ok=True)
        os.makedirs(test_data_folder + "/0", exist_ok=True)
        os.makedirs(test_data_folder + "/1", exist_ok=True)
        # 获取原始图片文件夹中的病例名列表
        case_names_0 = os.listdir(os.path.join(source_folder, "0"))
        case_names_1 = os.listdir(os.path.join(source_folder, "1"))
        print("===========================%s=========================="%name)
        print("%s_原始文件数量统计"%name,"0:%d,1:%d"%(len(case_names_0),len(case_names_1)),
            "img_0:%d,img_1:%d"%(count_images_in_folder(os.path.join(source_folder, "0"))
                                ,count_images_in_folder(os.path.join(source_folder, "1"))))
        # 随机打乱病例名顺序
        random.shuffle(case_names_0)
        random.shuffle(case_names_1)
        # 计算划分点
        split_point_0 = int(len(case_names_0) * train_ratio)
        split_point_1 = int(len(case_names_1) * train_ratio)
        count=0
        for i, case_name in enumerate(case_names_0):
            source_case_folder = os.path.join(source_folder, "0", case_name)
            if i < split_point_0:
                destination_case_folder = os.path.join(train_data_folder, "0", case_name)
                copy_images(source_case_folder, destination_case_folder)
                count+=1
                worksheet.write(count,0,case_name)
                worksheet.write(count,1,"train")
                worksheet.write(count,2,0)
            else:
                destination_case_folder = os.path.join(test_data_folder, "0", case_name)
                copy_images(source_case_folder, destination_case_folder)
                count+=1
                worksheet.write(count,0,case_name)
                worksheet.write(count,1,"test")
                worksheet.write(count,2,0)
        for i, case_name in enumerate(case_names_1):
            source_case_folder = os.path.join(source_folder, "1", case_name)
            if i < split_point_1:
                destination_case_folder = os.path.join(train_data_folder, "1", case_name)
                copy_images(source_case_folder, destination_case_folder)
                count+=1
                worksheet.write(count,0,case_name)
                worksheet.write(count,1,"train")
                worksheet.write(count,2,1)
            else:
                destination_case_folder = os.path.join(test_data_folder, "1", case_name)
                copy_images(source_case_folder, destination_case_folder)
                count+=1
                worksheet.write(count,0,case_name)
                worksheet.write(count,1,"test")
                worksheet.write(count,2,1)
        print("traindata,case_0:%d,case_1:%d,img_0:%d,img_1:%d"%(
            len(os.listdir(os.path.join(train_data_folder, "0"))),
            len(os.listdir(os.path.join(train_data_folder, "1"))),
            count_images_in_folder(os.path.join(train_data_folder, "0")),
            count_images_in_folder(os.path.join(train_data_folder, "1"))))
        print("testdata,case_0:%d,case_1:%d,img_0:%d,img_1:%d"%(
            len(os.listdir(os.path.join(test_data_folder, "0"))),
            len(os.listdir(os.path.join(test_data_folder, "1"))),
            count_images_in_folder(os.path.join(test_data_folder, "0")),
            count_images_in_folder(os.path.join(test_data_folder, "1"))))
    workbook.save(os.path.join("%s_%s_%d_%f"%(data_source,CT_type,random_seed,train_ratio),"patient_info.xls"))