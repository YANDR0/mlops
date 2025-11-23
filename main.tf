provider "aws" {
  region = "us-east-1"
}

# --- 1. SECURITY GROUP ---
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

# --- 2. GET UBUNTU IMAGE (Safe Method) ---
# This ensures you get a valid Free Tier AMI automatically
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- 3. THE INSTANCES ---
resource "aws_instance" "ml_node" {
  count         = 6
  ami           = data.aws_ami.ubuntu.id  # <--- Uses the safe AMI found above
  instance_type = "t3.micro"              # <--- MUST BE t2.micro FOR FREE TIER
  
  # !!! UPDATE THESE TWO LINES !!!
  key_name      = "mlproject"             # Name from AWS Console
  
  vpc_security_group_ids = [aws_security_group.ml_sg.id]
  user_data              = file("setup_script.sh")

  tags = {
    Name = "ML-Inference-Node-${count.index}"
  }

  # --- 4. UPLOAD & RUN ---
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("/Users/renata_temer/Downloads/mlproject.pem") 
    host        = self.public_ip
  }

  provisioner "file" {
    source      = "main.py"
    destination = "/home/ubuntu/app/main.py"
  }

  provisioner "file" {
    source      = "best_model.pth"
    destination = "/home/ubuntu/app/best_model.pth"
  }

  provisioner "remote-exec" {
    inline = [
      "sleep 120", # Wait longer for t2.micro to install everything
      "cd /home/ubuntu/app",
      "nohup uvicorn main:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &"
    ]
  }
}

output "node_ips" {
  value = aws_instance.ml_node[*].public_ip
}