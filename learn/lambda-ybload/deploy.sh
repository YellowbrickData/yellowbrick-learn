YBTOOLS_TGZ_NAME="ybtools-7.1.2-66379.e33141f2.generic.noarch.tar.gz"
YBLOAD_TGZ_NAME="${YBTOOLS_TGZ_NAME/ybtools/ybload}"

# S3 buckets for Lambda zip and parquet/csv files dropped in
ZIP_BUCKET=
LANDING_BUCKET=

# Change to suit your AWS account
ACCESS_KEY_ID=
SECRET_ACCESS_KEY=
SESSION_TOKEN=
REGION=
AWS_ACCOUNT=

# Yellowbrick database settings and target table
YB_HOST=
YB_USER=
YB_PASSWORD=
YB_DATABASE=
YB_TABLE=
YBLOAD_EXTRA_ARGS="--disable-trust"

# No need to change these unless a conflict
YBLOAD_FUNCTION="YBLoadFunction"
ROLE_NAME="lambda-s3-execution-role"
LAMBDA_JAVA_HOME="/var/lang"
JAR_NAME="ybload-lambda-1.0-SNAPSHOT.jar"
ZIP_NAME="ybload-lambda.zip"

./extract-ybload.sh "$YBTOOLS_TGZ_NAME"

mvn clean package || { echo "Maven build failed"; exit 1; }

rm -rf target/lambda_package
mkdir -p target/lambda_package

cd target/lambda_package
tar --strip-components=2 -xvf "../../$YBLOAD_TGZ_NAME" ybload/bin ybload/lib
jar -xvf "../$JAR_NAME"
zip -r "../$ZIP_NAME" *
cd ../..
aws s3 cp "target/$ZIP_NAME" s3://"$ZIP_BUCKET"/

ROLE_EXISTS=$(aws iam get-role --role-name "$ROLE_NAME" --query "Role.RoleName" --output text 2>/dev/null)
if [[ "$ROLE_EXISTS" == "$ROLE_NAME" ]]; then
    ATTACHED_POLICIES=$(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query "AttachedPolicies[*].PolicyArn" --output text)
    for POLICY_ARN in $ATTACHED_POLICIES; do
        echo "Detaching policy: $POLICY_ARN"
        aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN"
    done

    aws iam delete-role --role-name "$ROLE_NAME"
fi

aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'
aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

FUNCTION_EXISTS=$(aws lambda get-function --function-name "$YBLOAD_FUNCTION" --query "Configuration.FunctionName" --output text 2>/dev/null)
if [[ "$FUNCTION_EXISTS" == "$YBLOAD_FUNCTION" ]]; then
    aws lambda delete-function --function-name "$YBLOAD_FUNCTION"
fi

aws lambda create-function --no-cli-pager --function-name "$YBLOAD_FUNCTION" \
    --runtime java11 \
    --role arn:aws:iam::$AWS_ACCOUNT:role/lambda-s3-execution-role \
    --handler com.yellowbrick.ybload.YBLoadLambda::handleRequest \
    --code S3Bucket="$ZIP_BUCKET",S3Key="$ZIP_NAME" \
    --timeout 900 \
    --memory-size 10240

echo "Waiting for Lambda to become active..."
while true; do
    STATUS=$(aws lambda get-function --function-name "$YBLOAD_FUNCTION" --query "Configuration.State" --output text)
    if [ "$STATUS" == "Active" ]; then
        echo "Lambda is now active!"
        break
    fi
    echo "Lambda is in state: $STATUS... retrying in 5 seconds."
    sleep 5
done

aws lambda update-function-configuration --no-cli-pager --function-name "$YBLOAD_FUNCTION" \
    --environment "Variables={JAVA_HOME=\"$LAMBDA_JAVA_HOME\", \
    ACCESS_KEY_ID=\"$ACCESS_KEY_ID\", \
    SECRET_ACCESS_KEY=\"$(printf %q "$SECRET_ACCESS_KEY")\", \
    SESSION_TOKEN=\"$(printf %q "$SESSION_TOKEN")\", \
    YB_HOST=\"$YB_HOST\", \
    YB_USER=\"$YB_USER\", \
    YB_PASSWORD=\"$(printf %q "$YB_PASSWORD")\", \
    YB_DATABASE=\"$YB_DATABASE\", \
    YB_TABLE=\"$YB_TABLE\", \
    YBLOAD_EXTRA_ARGS=\"$YBLOAD_EXTRA_ARGS\"}"

aws lambda add-permission --function-name "$YBLOAD_FUNCTION" \
    --statement-id s3invoke \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::"$LANDING_BUCKET"

aws s3api put-bucket-notification-configuration --no-cli-pager --bucket "$LANDING_BUCKET" \
    --notification-configuration "{
        \"LambdaFunctionConfigurations\": [{
            \"LambdaFunctionArn\": \"arn:aws:lambda:$REGION:$AWS_ACCOUNT:function:$YBLOAD_FUNCTION\",
            \"Events\": [\"s3:ObjectCreated:*\"] 
        }]
    }"

echo "Deployment complete!"
