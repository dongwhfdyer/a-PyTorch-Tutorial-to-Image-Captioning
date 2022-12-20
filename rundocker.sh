image_name=pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9
#image_name=pytorch/pytorch:latest
sudo docker run -it --ipc=host --gpus all -v ~/kuhn/imcap:/workspace/imcap -v ~/kuhn/datasets:/workspace/datasets -v ~/kuhn/imcapdata:/workspace/imcapdata -p 9000:22 --name imcap_test $image_name