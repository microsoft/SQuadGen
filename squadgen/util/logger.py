import os
from torch.utils.tensorboard import SummaryWriter
import wandb

def convert_rgba_to_rgb(image):
    # image: [H, W, 4]
    # replace the transparent background with gray
    if image.shape[2] == 3:
        return image
    image = image.copy()
    image[image[:, :, 3] == 0] = [128, 128, 128, 255]
    return image[:, :, :3]

class Logger:
    wandb_project_name = None # set this to your wandb project name if you want to use wandb
    wandb_entity_name = None # set this to your wandb entity name if you want to use wandb

    def __init__(self, method, log_dir, name, resume) -> None:
        self.method = method # ["tensorboard", "wandb"]
        self.log_dir = log_dir
        self.name = name
        
        if self.method == "tensorboard":
            self.writer = SummaryWriter(log_dir)
        elif self.method == "wandb":
            if self.wandb_project_name is None or self.wandb_entity_name is None:
                raise ValueError("wandb_project_name and wandb_entity_name must be set to use wandb")
            if resume == '':
                print("Starting new wandb run")
                id = None
            else:
                folders = []
                if os.path.exists(os.path.join(log_dir, "wandb")):
                    for dirs in os.listdir(os.path.join(log_dir, "wandb")):
                        if dirs.startswith("run-"):
                            folders.append(dirs)
                if len(folders) > 0:
                    print(f"Resume wandb run: {folders[-1]}")
                    os.environ["WANDB_SILENT"] = "true"                
                    folders.sort()
                    id = "-".join(folders[-1].split("-")[2:])
                else:
                    print("Starting new wandb run")
                    id = None
            wandb.init(project=self.wandb_project_name, dir=log_dir, name=name, entity=self.wandb_entity_name,
                       resume="allow", id=id)

    def add_scalar(self, key, value, step):
        if self.method == "tensorboard":
            self.writer.add_scalar(key, value, step)
        elif self.method == "wandb":
            wandb.log({key: value}, step=step)

    def add_image(self, key, image, step, caption=None):
        if self.method == "tensorboard":
            self.writer.add_image(key, image, step, dataformats="HWC")
        elif self.method == "wandb":
            try:
                wandb.log({key: wandb.Image(convert_rgba_to_rgb(image), caption=caption, file_type="jpg")}, step=step)
            except Exception as e:
                print(f"[add_image] Error in adding image to wandb: {e}")

    def add_image_list(self, key, img_list, step):
        # key: [{"img": image1, "caption": caption1}, {"img": image2, "caption": caption2}, ...]
        if self.method == "tensorboard":
            for img_dict in img_list:
                self.writer.add_image(key, img_dict["img"], step, dataformats="HWC")
        elif self.method == "wandb":
            try:
                img_list = [wandb.Image(convert_rgba_to_rgb(img_dict["img"]), caption=img_dict["caption"], file_type="jpg") for img_dict in img_list]
                wandb.log({key: img_list}, step=step)
            except Exception as e:
                print(f"[add_image_list] Error in adding image to wandb: {e}")

    def log(self, log_dict, step):
        for key, value in log_dict.items():
            self.add_scalar(key, value, step)

    def close(self):
        if self.method == "tensorboard":
            self.writer.close()
        elif self.method == "wandb":
            wandb.finish()
    
    def flush(self):
        if self.method == "tensorboard":
            self.writer.flush()
        elif self.method == "wandb":
            pass