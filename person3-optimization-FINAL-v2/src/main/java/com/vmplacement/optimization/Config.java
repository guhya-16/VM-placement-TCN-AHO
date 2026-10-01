package com.vmplacement.optimization;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

public final class Config {
    private final Properties p;
    private Config(Properties p) { this.p = p; }

    public static Config load(Path path) throws IOException {
        Properties p = new Properties();
        try (InputStream in = Files.newInputStream(path)) { p.load(in); }
        return new Config(p);
    }

    public String get(String key) { return required(key); }
    public boolean bool(String key) { return Boolean.parseBoolean(required(key)); }
    public int integer(String key) { return Integer.parseInt(required(key)); }
    public long longValue(String key) { return Long.parseLong(required(key)); }
    public double decimal(String key) { return Double.parseDouble(required(key)); }
    public Path path(String key) { return Path.of(required(key)); }

    private String required(String key) {
        String v = p.getProperty(key);
        if (v == null || v.isBlank()) throw new IllegalArgumentException("Missing configuration: " + key);
        return v.trim();
    }
}
