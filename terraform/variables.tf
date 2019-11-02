variable "aws_profile" {
  type    = string
  default = "burner"
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "ec2_instance_type" {
  type    = string
  default = "p2.xlarge"
}

variable "ec2_ami" {
  type    = string
  default = "ami-0f9cf087c1f27d9b1"
}

variable "ec2_ssh_key_filepath" {
  type    = string
  default = "/Users/ivallesp/ssh_keys/burner.pem"
}

variable "aws_personal_credentials_filepath" {
  type    = string
  default = "/Users/ivallesp/.aws/credentials.ivallesp7"
}

variable "ec2_key_name" {
  type    = string
  default = "burner"
}

variable "kaggle_credentials_filepath" {
  type    = string
  default = "/Users/ivallesp/.kaggle"
}

variable "github_private_key_filepath" {
  type    = string
  default = "/Users/ivallesp/ssh_keys/cfavorita_github_key"
}

variable "amznet_prefix" {
  type    = list(string)
  default = ["pl-4e2ece27"]
}
