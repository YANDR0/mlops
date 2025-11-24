
provider "aws" {
  region = "us-east-1"
}

# --- SECURITY GROUP ---
resource "aws_security_group" "ml_sg" {
  name        = "ml_inference_sg"
  description = "Allow SSH and API traffic"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- UBUNTU IMAGE ---
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- 6 INSTANCES WITH API ---
resource "aws_instance" "ml_node" {
  count         = 6
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  key_name      = "mlproject"

  vpc_security_group_ids = [aws_security_group.ml_sg.id]

  tags = {
    Name = "ML-Inference-Node-${count.index}"
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("C:/Users/natha/Downloads/mlproject.pem")
    host        = self.public_ip
  }

  # Crear carpeta de la app
  provisioner "remote-exec" {
    inline = [
      "mkdir -p /home/ubuntu/app"
    ]
  }

  # Subir main.py
  provisioner "file" {
    source      = "main.py"
    destination = "/home/ubuntu/app/main.py"
  }

  # Subir modelo
  provisioner "file" {
    source      = "best_model.pth"
    destination = "/home/ubuntu/app/best_model.pth"
  }

  # ARRANCAR LA API
  provisioner "remote-exec" {
    inline = [
      "cd /home/ubuntu/app",
      "nohup uvicorn main:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &"
    ]
  }
}

output "node_ips" {
  value = aws_instance.ml_node[*].public_ip
}
