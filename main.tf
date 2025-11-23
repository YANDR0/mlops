provider "aws" {
  region = "us-east-1"
}

# --- 1. SECURITY GROUP (Actualizado: HTTP + ICMP) ---
resource "aws_security_group" "ml_sg" {
  name        = "ml_inference_sg_v2"
  description = "Allow SSH, API (5000), HTTP (80) and ICMP"

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # API Backend (Puerto 5000)
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP Frontend/Nginx (Puerto 80)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # ICMP (Ping)
  ingress {
    from_port   = -1
    to_port     = -1
    protocol    = "icmp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- 2. DATA SOURCE UBUNTU ---
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

# --- 3. BACKEND INSTANCES (6 Nodos - t3.small) ---
resource "aws_instance" "ml_node" {
  count         = 6
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.small"  # <--- Backend ahora es t3.small
  key_name      = "mlproject" 
  
  vpc_security_group_ids = [aws_security_group.ml_sg.id]
  user_data              = file("setup_script.sh")

  tags = {
    Name = "ML-Backend-Node-${count.index}"
    Role = "Backend"
  }

  # Provisión de archivos y arranque de API
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("C:/Users/migue/Downloads/mlproject.pem") 
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
      "sleep 120", 
      "cd /home/ubuntu/app",
      "nohup uvicorn main:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &"
    ]
  }
}

# --- 4. NGINX INSTANCE (1 Nodo - t3.micro - LIMPIA) ---
resource "aws_instance" "nginx_lb" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"
  key_name      = "mlproject"

  vpc_security_group_ids = [aws_security_group.ml_sg.id]
  
  # SIN user_data: Se configurará manualmente.
  
  tags = {
    Name = "ML-Nginx-LoadBalancer"
    Role = "LoadBalancer"
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("C:/Users/migue/Downloads/mlproject.pem")
    host        = self.public_ip
  }
}

# --- 5. FRONTEND INSTANCE (1 Nodo - t3.micro - LIMPIA) ---
resource "aws_instance" "frontend_app" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"
  key_name      = "mlproject"

  vpc_security_group_ids = [aws_security_group.ml_sg.id]
  
  tags = {
    Name = "ML-Frontend-App"
    Role = "Frontend"
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("C:/Users/migue/Downloads/mlproject.pem")
    host        = self.public_ip
  }
}

# --- 6. OUTPUTS ---
output "backend_ips" {
  description = "IPs Públicas de los 6 nodos Backend (t3.small)"
  value       = aws_instance.ml_node[*].public_ip
}

output "nginx_lb_ip" {
  description = "IP Pública para configurar Nginx manualmente."
  value       = aws_instance.nginx_lb.public_ip
}

output "frontend_ip" {
  description = "IP Pública de la instancia Frontend"
  value       = aws_instance.frontend_app.public_ip
}

# Output extra útil: IPs privadas de los backends para copiarlas a tu nginx.conf manual
output "backend_private_ips" {
  description = "IPs PRIVADAS de los backends (Úsalas en tu upstream de Nginx manual)"
  value       = aws_instance.ml_node[*].private_ip
}