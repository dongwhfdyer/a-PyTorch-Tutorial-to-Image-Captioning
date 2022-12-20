FROM pytorch/pytorch:latest
WORKDIR /workspace/imgcap
RUN apt-get update && apt install -y build-essential openssh-server vifm tmux vim git
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN cat /etc/ssh/sshd_config | sed 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' > /etc/ssh/sshd_config