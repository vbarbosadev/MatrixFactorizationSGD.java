package SGD;

import com.google.gson.Gson;
import com.google.gson.stream.JsonWriter;
import com.google.gson.reflect.TypeToken;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class MovieRecommenderSGD {




    public static class Rating {
        String user_id;
        String title;
        List<String> genres;
        double rating;

        public Rating(String user_id, String title, List<String> genres, double rating) {
            this.user_id = user_id;
            this.title = title;
            this.genres = genres;
            this.rating = rating;
        }

        public String getUserId() { return user_id; }
        public String getTitle() { return title; }
        public List<String> getGenres() { return genres; }
        public double getRating() { return rating; }
    }

    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    public static Map<String, double[]> userFactors = new HashMap<>();
    public static Map<String, double[]> genreFactors = new HashMap<>();

    public static Set<String> allGenres = new HashSet<>();
    public static Set<String> allUsers = new HashSet<>();

    @SuppressWarnings("unused")
    public static void main(String[] args) throws Exception {

        long startTime = System.nanoTime();

        List<Rating> ratings = loadRatings("dataset/avaliacoes_divididas/avaliacoes_parte_1.json");
        //List<Rating> ratings = loadRatings("dataset/avaliacoes_completas100MB.json");

        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);




        Map<String, List<String>> movieToGenresMap = new HashMap<>();
        for (Rating r : ratings) {
            allUsers.add(r.getUserId());
            allGenres.addAll(r.getGenres());
            movieToGenresMap.putIfAbsent(r.getTitle(), r.getGenres());
        }

        long sortTime = System.nanoTime();
        System.out.printf("reorganizado em: %.2f segundos%n", (sortTime - readTime) / 1e9);


        readTime = System.nanoTime();

        initializeFactors();

        long initialTime = System.nanoTime();
        System.out.printf("initializeFactors rodou em: %.2f segundos%n", (initialTime - readTime) / 1e9);




        trainModel(ratings);

        long trainingTime = System.nanoTime();
        System.out.printf("trainingModel rodou em: %.2f segundos%n", (trainingTime - initialTime) / 1e9);


       Map<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings, movieToGenresMap);


        ratings = null;

        userFactors = null;
        genreFactors = null;
        allGenres = null;


        long matrixTime = System.nanoTime();
        System.out.printf("matrixGen rodou em: %.2f segundos%n", (matrixTime - trainingTime) / 1e9);

        long printTime = System.nanoTime();

        String outputDir = "output_ratings";
        new java.io.File(outputDir).mkdirs();


        savePredictedRatingsToJson(matrix, allUsers, movieToGenresMap, "dataset/predicted_ratings.json");

        long saveTime = System.nanoTime();
        System.out.printf("saveJson rodou em: %.2f segundos%n", (saveTime - printTime) / 1e9);


        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);


    }

    public static List<Rating> loadRatings(String filename) throws IOException {
        Gson gson = new Gson();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            return gson.fromJson(reader, new TypeToken<List<Rating>>() {}.getType());
        }
    }

    public static void initializeFactors() {
        Random rand = new Random();
        for (String user : allUsers) {
            userFactors.put(user, rand.doubles(NUM_FEATURES, 0, 0.1).toArray());
        }
        for (String genre : allGenres) {
            genreFactors.put(genre, rand.doubles(NUM_FEATURES, 0, 0.1).toArray());
        }
    }

    public static void trainModel(List<Rating> ratings) {
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            for (Rating r : ratings) {
                String user = r.getUserId();
                double[] userVec = userFactors.get(user);

                for (String genre : r.getGenres()) {
                    double[] genreVec = genreFactors.get(genre);
                    double prediction = dot(userVec, genreVec);
                    double error = r.getRating() - prediction;

                    for (int i = 0; i < NUM_FEATURES; i++) {
                        double u = userVec[i];
                        double g = genreVec[i];
                        userVec[i] += LEARNING_RATE * (error * g - REGULARIZATION * u);
                        genreVec[i] += LEARNING_RATE * (error * u - REGULARIZATION * g);
                    }
                }
            }
        }
    }

    public static Map<String, Map<String, Double>> predictRatingsMatrix(List<MovieRecommenderSGD.Rating> ratings, Map<String, List<String>> movieToGenresMap) {
        Map<String, Map<String, Double>> matrix = new LinkedHashMap<>();
        Map<String, Map<String, Double>> userRatedMovies = new HashMap<>();
        for (MovieRecommenderSGD.Rating r : ratings) {
            userRatedMovies.computeIfAbsent(r.getUserId(), k -> new HashMap<>()).put(r.getTitle(), r.getRating());
        }

        long predictionCounter = 0;

        for (String user : allUsers) {
            Map<String, Double> ratingsForUser = new LinkedHashMap<>();
            double[] userVec = userFactors.get(user);
            Map<String, Double> existingRatings = userRatedMovies.getOrDefault(user, Collections.emptyMap());

            for (Map.Entry<String, List<String>> movieEntry : movieToGenresMap.entrySet()) {
                String title = movieEntry.getKey();
                if (existingRatings.containsKey(title)) {
                    ratingsForUser.put(title, existingRatings.get(title));
                } else {
                    predictionCounter++;

                    List<String> genres = movieEntry.getValue();
                    double predicted = 0.0;
                    if (genres != null && !genres.isEmpty()) {
                        for (String genre : genres) {
                            predicted += dot(userVec, genreFactors.get(genre));
                        }
                        predicted /= genres.size();
                    }
                    ratingsForUser.put(title, predicted);
                }
            }
            matrix.put(user, ratingsForUser);
        }

        System.out.println("\n--------------------------------------------------");
        System.out.println("Total de previsões (predicts) realizadas: " + predictionCounter);
        System.out.println("--------------------------------------------------\n");

        return matrix;
    }


    public static void savePredictedRatingsToJson(Map<String, Map<String, Double>> predictedMatrix,
                                                  Set<String> users,
                                                  Map<String, List<String>> genreMap,
                                                  String filename) {
        List<String> allUsersOrdered = new ArrayList<>(users);
        Collections.sort(allUsersOrdered);

        List<String> allTitlesOrdered = new ArrayList<>(genreMap.keySet());
        Collections.sort(allTitlesOrdered);

        try (JsonWriter writer = new JsonWriter(new FileWriter(filename))) {
            writer.setIndent("  ");

            writer.beginArray();

            for (String user : allUsersOrdered) {
                writer.beginObject();
                writer.name("user_id").value(user);

                writer.name("movies");
                writer.beginArray();

                Map<String, Double> userRatings = predictedMatrix.getOrDefault(user, Collections.emptyMap());

                for (String title : allTitlesOrdered) {
                    writer.beginObject();
                    writer.name("title").value(title);

                    writer.name("genre");
                    writer.beginArray();
                    List<String> genres = genreMap.getOrDefault(title, Collections.emptyList());
                    for (String genre : genres) {
                        writer.value(genre);
                    }
                    writer.endArray();

                    writer.name("rating");
                    Double rating = userRatings.get(title);
                    if (rating != null) {
                        writer.value(rating);
                    } else {
                        writer.nullValue();
                    }
                    writer.endObject();
                }
                writer.endArray();
                writer.endObject();
            }
            writer.endArray();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}