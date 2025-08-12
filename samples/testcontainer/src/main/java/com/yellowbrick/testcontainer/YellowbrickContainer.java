package com.yellowbrick.testcontainer;

import com.github.dockerjava.api.command.InspectContainerResponse;
import org.testcontainers.containers.Container;
import org.testcontainers.containers.JdbcDatabaseContainer;
import org.testcontainers.containers.wait.strategy.LogMessageWaitStrategy;
import org.testcontainers.utility.DockerImageName;
import org.testcontainers.utility.MountableFile;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Set;

/**
 * Testcontainers implementation for Yellowbrick Database Community Edition.
 *
 * <p>This container provides a test instance of Yellowbrick database that can be used
 * for integration testing. It supports automatic initialization with SQL scripts,
 * bootstrap data copying, and configurable connection parameters.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * @Container
 * static YellowbrickContainer yellowbrick = YellowbrickContainer.create()
 *     .withDatabaseName("testdb")
 *     .withUsername("testuser")
 *     .withPassword("testpass")
 *     .withBootstrapData("test/data")
 *     .withMemory(16)
 *     .withCpuCount(8)
 *     .withDebugMode(true);
 * }</pre>
 *
 * @author Yellowbrick Spring AI Team
 */
public class YellowbrickContainer extends JdbcDatabaseContainer<YellowbrickContainer> {

    public static final String IMAGE = "yellowbrickdata/yb-community-edition";
    public static final String DEFAULT_TAG = "latest";

    // Default connection parameters - use builder methods to override
    public static final String DEFAULT_USER = "ybdadmin";
    public static final String DEFAULT_PASSWORD = "ybdadmin";
    public static final String DEFAULT_DATABASE_NAME = "yellowbrick";

    // Default resource limits - use builder methods to override
    public static final long DEFAULT_MEMORY_GB = 8L;
    public static final long DEFAULT_CPU_COUNT = 4L;
    public static final String DEFAULT_IPC_MODE = "private";

    // Default timeouts - use builder methods to override
    public static final int DEFAULT_STARTUP_TIMEOUT_MINUTES = 10;
    public static final int DEFAULT_READY_TIMEOUT_MINUTES = 5;

    static final Integer YELLOWBRICK_PORT = 5432;
    static final Integer HTTPS_PORT = 443;

    private String databaseName = DEFAULT_DATABASE_NAME;
    private String username = DEFAULT_USER;
    private String password = DEFAULT_PASSWORD;
    private String bootstrapDataPath = null;
    private boolean debugMode = false;

    /**
     * Create YellowbrickContainer with default image and latest tag.
     */
    public YellowbrickContainer() {
        this(IMAGE + ":" + DEFAULT_TAG);
    }

    /**
     * Create YellowbrickContainer with specific docker image name.
     *
     * @param dockerImageName the Docker image name (e.g., "yellowbrickdata/yb-community-edition:latest")
     */
    public YellowbrickContainer(String dockerImageName) {
        this(DockerImageName.parse(dockerImageName).asCompatibleSubstituteFor(IMAGE));
    }

    /**
     * Create YellowbrickContainer with DockerImageName.
     *
     * @param dockerImageName the Docker image name as DockerImageName object
     */
    public YellowbrickContainer(DockerImageName dockerImageName) {
        super(dockerImageName);

        this.waitStrategy = new LogMessageWaitStrategy()
                .withRegEx(".*Connect to ybsql as user ybdadmin.*")
                .withTimes(1)
                .withStartupTimeout(Duration.of(DEFAULT_STARTUP_TIMEOUT_MINUTES, ChronoUnit.MINUTES));

        configure();
    }

    /**
     * Configure the container with default settings.
     * This method is protected to allow subclasses to override configuration behavior.
     */
    protected void configure() {
        // Expose both PostgreSQL and HTTPS ports
        addExposedPorts(YELLOWBRICK_PORT, HTTPS_PORT);

        // Yellowbrick requires privileged mode for proper operation
        withPrivilegedMode(true);

        copyBootstrapData();

        // Set Yellowbrick debug environment variable
        withEnv("YB_DEBUG", String.valueOf(debugMode));

        // Set resource limits and IPC mode
        withCreateContainerCmdModifier(cmd -> {
            cmd.getHostConfig()
                    .withIpcMode(DEFAULT_IPC_MODE)
                    .withMemory(DEFAULT_MEMORY_GB * 1024 * 1024 * 1024L) // Convert GB to bytes
                    .withCpuCount(DEFAULT_CPU_COUNT);
        });

        // Set startup timeout
        withStartupTimeout(Duration.of(DEFAULT_STARTUP_TIMEOUT_MINUTES, ChronoUnit.MINUTES));
    }

    /**
     * Copy bootstrap data from classpath resource to container.
     * This method will silently skip if the resource path is null or doesn't exist.
     *
     * @param classpathResourcePath the classpath resource path (e.g., "test/data" or "yellowbrick/sql/init.sql")
     * @return this container instance for method chaining
     */
    public YellowbrickContainer withBootstrapData(String classpathResourcePath) {
        if (classpathResourcePath == null || classpathResourcePath.trim().isEmpty()) {
            logger().debug("Bootstrap data path is null or empty, skipping bootstrap data copy");
            return self();
        }

        this.bootstrapDataPath = classpathResourcePath.trim();
        return self();
    }

    /**
     * Copy bootstrap data files to the container if configured.
     * This is called automatically during container startup.
     */
    private void copyBootstrapData() {
        if (bootstrapDataPath == null) {
            logger().debug("No bootstrap data path configured, skipping bootstrap data copy");
            return;
        }

        try {
            // Check if the resource exists
            ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
            java.net.URL resourceUrl = classLoader.getResource(bootstrapDataPath);

            if (resourceUrl == null) {
                logger().warn("Bootstrap data resource not found at path: {}, skipping copy", bootstrapDataPath);
                return;
            }

            logger().info("Found bootstrap data at: {}, copying to container /mnt/bootstrap/", bootstrapDataPath);

            // Determine target path based on whether it's a single file or directory
            String targetPath;
            if (bootstrapDataPath.contains(".")) {
                // Likely a single file, preserve the filename
                String fileName = bootstrapDataPath.substring(bootstrapDataPath.lastIndexOf("/") + 1);
                targetPath = "/mnt/bootstrap/" + fileName;
            } else {
                // Directory, copy to /mnt/bootstrap
                targetPath = "/mnt/bootstrap";
            }

            withCopyToContainer(MountableFile.forClasspathResource(bootstrapDataPath), targetPath);
            logger().info("Bootstrap data successfully copied to: {}", targetPath);

        } catch (Exception e) {
            logger().warn("Failed to copy bootstrap data from {}: {}", bootstrapDataPath, e.getMessage());
            // Don't fail container creation if bootstrap data copy fails
        }
    }

    @Override
    protected void containerIsStarted(InspectContainerResponse containerInfo) {
        super.containerIsStarted(containerInfo);
        // Copy bootstrap data after container starts but before it's marked as ready
    }

    @Override
    protected Set<Integer> getLivenessCheckPorts() {
        return Set.of(getMappedPort(YELLOWBRICK_PORT));
    }

    @Override
    public String getDriverClassName() {
        return "org.postgresql.Driver";
    }

    @Override
    public String getJdbcUrl() {
        String additionalUrlParams = constructUrlParameters("?", "&");
        return "jdbc:postgresql://" + getHost() + ":" + getMappedPort(YELLOWBRICK_PORT)
                + "/" + databaseName + additionalUrlParams;
    }

    @Override
    public String getDatabaseName() {
        return databaseName;
    }

    @Override
    public String getUsername() {
        return username;
    }

    @Override
    public String getPassword() {
        return password;
    }

    @Override
    protected String getTestQueryString() {
        return "SELECT 1";
    }

    @Override
    public YellowbrickContainer withDatabaseName(String databaseName) {
        this.databaseName = databaseName != null ? databaseName : DEFAULT_DATABASE_NAME;
        return self();
    }

    @Override
    public YellowbrickContainer withUsername(String username) {
        this.username = username != null ? username : DEFAULT_USER;
        return self();
    }

    @Override
    public YellowbrickContainer withPassword(String password) {
        this.password = password != null ? password : DEFAULT_PASSWORD;
        return self();
    }


    /**
     * Enable or disable Yellowbrick debug mode.
     *
     * @param enabled true to enable debug mode, false to disable
     * @return this container instance for method chaining
     */
    public YellowbrickContainer withDebugMode(boolean enabled) {
        this.debugMode = enabled;
        withEnv("YB_DEBUG", String.valueOf(enabled));
        return self();
    }

    /**
     * Set the IPC mode for the container.
     *
     * @param ipcMode the IPC mode (e.g., "host", "private", "none", "shareable")
     * @return this container instance for method chaining
     */
    public YellowbrickContainer withIpcMode(String ipcMode) {
        withCreateContainerCmdModifier(cmd -> cmd.getHostConfig().withIpcMode(ipcMode));
        return self();
    }

    /**
     * Set memory limit for the container.
     *
     * @param memoryGb memory limit in GB
     * @return this container instance for method chaining
     */
    public YellowbrickContainer withMemory(long memoryGb) {
        withCreateContainerCmdModifier(cmd -> cmd.getHostConfig().withMemory(memoryGb * 1024 * 1024 * 1024L));
        return self();
    }

    /**
     * Set CPU count for the container.
     *
     * @param cpuCount number of CPU cores
     * @return this container instance for method chaining
     */
    public YellowbrickContainer withCpuCount(long cpuCount) {
        withCreateContainerCmdModifier(cmd -> cmd.getHostConfig().withCpuCount(cpuCount));
        return self();
    }

    /**
     * Set startup timeout for the container.
     *
     * @param timeoutMinutes timeout in minutes
     * @return this container instance for method chaining
     */
    public YellowbrickContainer withStartupTimeout(int timeoutMinutes) {
        withStartupTimeout(Duration.of(timeoutMinutes, ChronoUnit.MINUTES));
        return self();
    }

    /**
     * Set startup timeout for the container.
     *
     * @param timeout timeout duration
     * @return this container instance for method chaining
     */
    public YellowbrickContainer withStartupTimeout(Duration timeout) {
        super.withStartupTimeout(timeout);
        return self();
    }

    /**
     * Get the mapped port for HTTPS access to Yellowbrick web interface.
     *
     * @return the mapped HTTPS port
     */
    public Integer getHttpsPort() {
        return getMappedPort(HTTPS_PORT);
    }

    /**
     * Get the URL for Yellowbrick web interface.
     *
     * @return the web interface URL
     */
    public String getWebInterfaceUrl() {
        return "https://" + getHost() + ":" + getHttpsPort();
    }

    /**
     * Execute ybsql command inside the container with proper environment variables.
     *
     * @param command the ybsql command arguments
     * @return the execution result
     * @throws Exception if command execution fails
     */
    public Container.ExecResult executeYbsqlCommand(String... command) throws Exception {
        // Build command with environment variables
        String[] envCommand = new String[command.length + 5];
        envCommand[0] = "env";
        envCommand[1] = "YBPASSWORD=" + password;
        envCommand[2] = "YBDATABASE=" + databaseName;
        envCommand[3] = "YBUSER=" + username;
        envCommand[4] = "ybsql";
        System.arraycopy(command, 0, envCommand, 5, command.length);

        return execInContainer(envCommand);
    }

    /**
     * Execute SQL query using ybsql.
     *
     * @param sql the SQL query to execute
     * @return the execution result
     * @throws Exception if query execution fails
     */
    public Container.ExecResult executeQuery(String sql) throws Exception {
        return executeYbsqlCommand("-c", sql);
    }

    /**
     * Check if Yellowbrick is ready by connecting with ybsql and checking cluster state.
     *
     * @return true if Yellowbrick is ready, false otherwise
     */
    public boolean isYellowbrickReady() {
        try {
            Container.ExecResult result = executeQuery("SELECT state FROM sys.cluster;");
            boolean isReady = result.getExitCode() == 0 && result.getStdout().contains("RUNNING");
            logger().info("Yellowbrick readiness check - Exit code: {}, Contains RUNNING: {}",
                    result.getExitCode(), result.getStdout().contains("RUNNING"));
            return isReady;
        } catch (Exception e) {
            logger().info("Yellowbrick readiness check failed: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Wait for Yellowbrick to be fully ready for connections with default timeout.
     */
    public void waitUntilYellowbrickReady() {
        waitUntilYellowbrickReady(Duration.of(DEFAULT_READY_TIMEOUT_MINUTES, ChronoUnit.MINUTES));
    }

    /**
     * Wait for Yellowbrick to be ready with custom timeout.
     *
     * @param timeout the maximum time to wait
     * @throws RuntimeException if Yellowbrick doesn't become ready within the timeout
     */
    public void waitUntilYellowbrickReady(Duration timeout) {
        long startTime = System.currentTimeMillis();
        long timeoutMillis = timeout.toMillis();
        int attemptCount = 0;

        logger().info("Waiting for Yellowbrick to be ready (timeout: {})", timeout);

        while (System.currentTimeMillis() - startTime < timeoutMillis) {
            attemptCount++;
            if (isYellowbrickReady()) {
                logger().info("Yellowbrick is ready for connections after {} attempts", attemptCount);
                return;
            }

            logger().debug("Yellowbrick readiness check attempt {} failed, retrying in 5 seconds...", attemptCount);
            try {
                Thread.sleep(5000); // Check every 5 seconds
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while waiting for Yellowbrick to be ready", e);
            }
        }

        throw new RuntimeException(String.format(
                "Yellowbrick did not become ready within %s after %d attempts", timeout, attemptCount));
    }

    /**
     * Create a YellowbrickContainer with default configuration.
     *
     * @return a new YellowbrickContainer instance
     */
    public static YellowbrickContainer create() {
        return new YellowbrickContainer();
    }

    /**
     * Create a YellowbrickContainer with specific tag.
     *
     * @param tag the Docker image tag
     * @return a new YellowbrickContainer instance
     */
    public static YellowbrickContainer create(String tag) {
        return new YellowbrickContainer(IMAGE + ":" + tag);
    }

    /**
     * Create a YellowbrickContainer with specific image.
     *
     * @param dockerImageName the full Docker image name
     * @return a new YellowbrickContainer instance
     */
    public static YellowbrickContainer createWithImage(String dockerImageName) {
        return new YellowbrickContainer(dockerImageName);
    }
}