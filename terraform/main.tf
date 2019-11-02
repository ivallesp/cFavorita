provider "aws" {
  profile = var.aws_profile
  region  = var.aws_region
}

resource "aws_security_group" "ssh-from-amznet" {
  name        = "ssh-from-amznet"
  description = "Access to SSH from amazon network"
  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    prefix_list_ids = var.amznet_prefix
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "example" {
  ami                  = var.ec2_ami
  instance_type        = var.ec2_instance_type
  security_groups      = [aws_security_group.ssh-from-amznet.name]
  iam_instance_profile = aws_iam_instance_profile.test_profile.name
  key_name             = var.ec2_key_name

  connection {
    host        = aws_instance.example.public_dns
    user        = "ubuntu"
    private_key = file(var.ec2_ssh_key_filepath)
    agent       = true
    timeout     = "3m"
  }

  provisioner "remote-exec" {
    inline = [<<EOF
              mkdir ~/ssh_keys
              mkdir ~/.aws
              EOF
            ]
  }

  provisioner "file" {
    source = var.github_private_key_filepath
    destination = "/home/ubuntu/ssh_keys/cfavorita_github_key"
  }

  provisioner "file" {
    source = var.kaggle_credentials_filepath
    destination = "/home/ubuntu/"
  }

  provisioner "file" {
    source = var.aws_personal_credentials_filepath
    destination = "~/.aws/credentials"
  }

  provisioner "remote-exec" {
    inline = [<<EOF
              sudo fuser -vk  /var/lib/dpkg/lock
              sudo rm /var/lib/dpkg/lock
              sudo dpkg --configure -a
              sudo apt update
              sudo apt --yes install p7zip-full libffi-dev python python-pip make \
              build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
              libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils \
              tk-dev libffi-dev liblzma-dev python-openssl git awscli unzip htop

              # Install pyenv dependencies
              curl https://pyenv.run | bash
              echo 'export PATH="/home/ubuntu/.pyenv/bin:$PATH"' >> ~/.bashrc
              echo 'eval "$(pyenv init -)"' >> ~/.bashrc
              eval "$(~/.pyenv/bin/pyenv init -)"

              cd .

              # Install poetry
              curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
              echo 'source $HOME/.poetry/env' >> ~/.bashrc
              ~/.poetry/bin/poetry config settings.virtualenvs.in-project true
              EOF
    ]
  }
  provisioner "local-exec" {
    command = "sed -i.bk '/^Host burner/,/^\\[/{s/^Hostname.*/Hostname ${aws_instance.example.public_dns}/}' ~/.ssh/config"
  }
}

resource "aws_ebs_volume" "datadrive" {
  availability_zone = aws_instance.example.availability_zone
  size              = 1000
}

resource "aws_volume_attachment" "ebs_data_att" {
  device_name  = "/dev/sdh"
  volume_id    = aws_ebs_volume.datadrive.id
  instance_id  = aws_instance.example.id
  force_detach = true
  provisioner "remote-exec" {
    connection {
      host        = aws_instance.example.public_dns
      user        = "ubuntu"
      private_key = file(var.ec2_ssh_key_filepath)
      agent       = true
      timeout     = "3m"
    }
    inline = [<<EOF
              source ~/.bashrc

	            # Set data volume
	            sudo mkfs -t ext4 /dev/xvdh
	            sudo mkdir /data
	            sudo mount /dev/xvdh /data
	            sudo chmod 777 /data
              chmod 600 /home/ubuntu/.kaggle/kaggle.json
	            cd /data
	            echo "${aws_s3_bucket.main.bucket}" > s3_bucket
	            mkdir .logs_tensorboard

	            # Clone repository
	            echo 'Host github.com'>> ~/.ssh/config
	            echo '     StrictHostKeyChecking no' >> ~/.ssh/config
              eval `ssh-agent -s`
              chmod 400 ~/ssh_keys/cfavorita_github_key
	            ssh-add ~/ssh_keys/cfavorita_github_key
	            git clone git@github.com:ivallesp/cFavorita.git
              cd cFavorita
              git remote rm origin
              git remote add origin https://github.com/ivallesp/cFavorita.git

              # S3 sync
              aws s3 sync s3://phd-cfavorita/models /data/cFavorita/models
              aws s3 sync s3://phd-cfavorita/logs_tensorboard /data/.logs_tensorboard
              (crontab -l ; echo "*/5 * * * * /usr/local/bin/aws s3 sync /data/cFavorita/models s3://phd-cfavorita/models") | sort - | uniq - | crontab -
              (crontab -l ; echo "*/30 * * * * /usr/local/bin/aws s3 sync /data/.logs_tensorboard s3://phd-cfavorita/logs_tensorboard") | sort - | uniq - | crontab -

              echo '{ "paths": {"data":  "data", "tensorboard": "/data/.logs_tensorboard"}}' > ./settings.json

              # Set up the environment
              eval "$(~/.pyenv/bin/pyenv init -)"
              ~/.pyenv/bin/pyenv install "$(cat .python-version)"
              ~/.pyenv/bin/pyenv local "$(cat .python-version)"
              cd .
              ~/.poetry/bin/poetry install

              # Download the data
	            mkdir data
	            cd data
	            /data/cFavorita/.venv/bin/kaggle competitions download -c favorita-grocery-sales-forecasting
              unzip *.zip
	            7za -y x "*.7z"

              # Setup tmux
              cd /data/cFavorita
              tmux new -d -s foo -n jupyter
              tmux send-keys -t foo "jupyter notebook" ENTER
              tmux send-keys -t foo "source ./.venv/bin/activate" ENTER
              tmux new-window -t foo -n tensorboard
              tmux send-keys -t foo "source ./.venv/bin/activate" ENTER
              tmux send-keys -t foo "tensorboard --logdir /data/.logs_tensorboard" ENTER
              tmux new-window -t foo -n htop
              tmux send-keys -t foo "htop" ENTER
              tmux new-window -t foo -n nvidia-smi
              tmux send-keys -t foo "watch nvidia-smi" ENTER
              tmux new-window -t foo -n RUN
              tmux send-keys -t foo "source ./.venv/bin/activate" ENTER
              tmux send-keys -t foo "python main.py" ENTER
              EOF
    ]
  }
}

resource "aws_ebs_volume" "swapdrive" {
  availability_zone = aws_instance.example.availability_zone
  type = "gp2"
  size = 250
}

resource "aws_volume_attachment" "ebs_swap_att" {
  device_name = "/dev/sds"
  volume_id = aws_ebs_volume.swapdrive.id
  instance_id = aws_instance.example.id
  force_detach = true
  provisioner "remote-exec" {
    connection {
      host = aws_instance.example.public_dns
      user = "ubuntu"
      private_key = file(var.ec2_ssh_key_filepath)
      agent = true
      timeout = "3m"
    }
    inline = [<<EOF
      	  # Set SWAP
	      sudo mkswap /dev/xvds
	      sudo swapon /dev/xvds
       EOF
    ]
  }
}

resource "aws_s3_bucket" "main" {
  acl = "private"
  tags = {
    Name        = "My bucket"
    Environment = "Dev"
  }
}

output "ip_address" {
  value = aws_instance.example.public_dns
}

output "s3_bucket" {
  value = aws_s3_bucket.main.bucket
}
