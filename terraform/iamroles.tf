resource "aws_iam_role" "ec2_s3_access_role" {
  name               = "my_s3-role"
  assume_role_policy = file("assumerolepolicy.json")
}

resource "aws_iam_policy" "policy" {
  name        = "my-policy"
  description = "A test policy"
  policy      = file("policys3bucket.json")
}

resource "aws_iam_policy_attachment" "test-attach" {
  name       = "my_policy_attachment"
  roles      = [aws_iam_role.ec2_s3_access_role.name]
  policy_arn = aws_iam_policy.policy.arn
}

resource "aws_iam_instance_profile" "test_profile" {
  name  = "my_profile"
  roles = [aws_iam_role.ec2_s3_access_role.name]
}
