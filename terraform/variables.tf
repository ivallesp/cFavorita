variable "aws_profile" {
  type    = string
  default = "burner"
}

variable "n_ec2_instances" {
  type    = number
  default = 1
}

variable "ec2_public_key_filepath" {
  type    = string
  default = "~/ssh_keys/burner.pub"
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
  default = "ami-09ccc07b3d5df2bc6"
}

variable "ec2_ssh_key_filepath" {
  type    = string
  default = "~/ssh_keys/burner.pem"
}

variable "aws_personal_credentials_filepath" {
  type    = string
  default = "~/.aws/credentials.ivallesp7"
}

variable "ec2_key_name" {
  type    = string
  default = "burner"
}

variable "kaggle_credentials_filepath" {
  type    = string
  default = "~/.kaggle"
}

variable "github_private_key_filepath" {
  type    = string
  default = "~/ssh_keys/cfavorita_github_key"
}

variable "amznet_prefix" {
  type    = list(string)
  default = ["pl-4e2ece27"]
}
