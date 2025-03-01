package com.yellowbrick.ybload;

import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.events.models.s3.S3EventNotification;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class YBLoadLambda {
    public void handleRequest(S3EventNotification event, Context context) {
        List<S3EventNotification.S3EventNotificationRecord> records = event.getRecords();

        for (S3EventNotification.S3EventNotificationRecord record : records) {
            String bucketName = record.getS3().getBucket().getName();
            String objectKey = record.getS3().getObject().getKey();

            context.getLogger().log("Processing file: " + objectKey + " from bucket: " + bucketName);

            String format;
            if (objectKey.endsWith(".parquet")) {
                format = "parquet";
            } else if (objectKey.endsWith(".csv") || objectKey.endsWith(".csv.gz")) {
                format = "csv";
            } else {
                format = "unknown";
            }

            try {
                String accessKey = System.getenv("ACCESS_KEY_ID");
                String secretKey = System.getenv("SECRET_ACCESS_KEY");
                String sessionToken = System.getenv("SESSION_TOKEN");
                String javaHome = System.getenv("JAVA_HOME");
                String ybHost = System.getenv("YB_HOST");
                String ybUser = System.getenv("YB_USER");
                String ybPassword = System.getenv("YB_PASSWORD");
                String ybDatabase = System.getenv("YB_DATABASE");
                String ybTable = System.getenv("YB_TABLE");
                String ybloadExtraArgs = System.getenv("YBLOAD_EXTRA_ARGS");


                if (accessKey == null || secretKey == null || sessionToken == null) {
                    throw new RuntimeException("Missing AWS credentials in environment variables.");
                }

                Map<String, String> env = new HashMap<>(System.getenv());
                env.put("AWS_ACCESS_KEY_ID", accessKey);
                env.put("AWS_SECRET_ACCESS_KEY", secretKey);
                env.put("AWS_SESSION_TOKEN", sessionToken);
                env.put("JAVA_HOME", javaHome);
                env.put("YBPASSWORD", ybPassword);

                List<String> command = new ArrayList<>(Arrays.asList(
                    "/var/task/bin/ybload",
                    "-d", ybDatabase,
                    "-h", ybHost,
                    "-U", ybUser,
                    "-t", ybTable,
                    "--format", format,
                    "--bad-row-file", "/dev/stdout"
                ));

                if (ybloadExtraArgs != null && !ybloadExtraArgs.trim().isEmpty()) {
                    command.addAll(Arrays.asList(ybloadExtraArgs.trim().split("\\s+")));
                }

                command.add("s3://" + bucketName + "/" + objectKey);

                System.out.println("Executing command: " + String.join(" ", command));

                ProcessBuilder pb = new ProcessBuilder(command);
                pb.redirectErrorStream(true);
                pb.environment().putAll(env);

                Process process = pb.start();

                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String line;
                while ((line = reader.readLine()) != null) {
                    context.getLogger().log(line);
                }

                int exitCode = process.waitFor();
                context.getLogger().log("ybload completed with exit code: " + exitCode);

            } catch (Exception e) {
                context.getLogger().log("Error executing ybload: " + e.getMessage());
            }
        }
    }
}
