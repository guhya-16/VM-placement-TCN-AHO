package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;

final class Csv {
    private Csv() {}

    static List<String> parseLine(String line) {
        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (quoted && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"'); i++;
                } else quoted = !quoted;
            } else if (c == ',' && !quoted) {
                values.add(current.toString().trim()); current.setLength(0);
            } else current.append(c);
        }
        values.add(current.toString().trim());
        return values;
    }

    static String escape(String value) {
        if (value == null) return "";
        String v = value.replace("\"", "\"\"");
        return (v.indexOf(',') >= 0 || v.indexOf('\"') >= 0) ? "\"" + v + "\"" : v;
    }

    static String require(List<String> row, java.util.Map<String,Integer> header, String name) {
        Integer i = header.get(name);
        if (i == null || i >= row.size()) throw new IllegalArgumentException("Missing required CSV field: " + name);
        return row.get(i);
    }
}
