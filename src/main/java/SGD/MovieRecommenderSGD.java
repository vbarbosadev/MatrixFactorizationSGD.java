package SGD;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;


public class MovieRecommenderSGD {

    private static class Rating {
        String user_id;
        String title;
        List<String> genre;
        double rating;

        public Rating(String user_id, String title, List<String> genre, double rating) {
            this.user_id = user_id;
            this.title = title;
            this.genre = genre;
            this.rating = rating;
        }

        public String getUserId() {
            return user_id;
        }

        public String getTitle() {
            return title;
        }

        public List<String> getGenres() {
            return genre;
        }

        public double getRating() {
            return rating;
        }
    }

    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    private static Map<String, double[]> userFactors = new HashMap<>();
    private static Map<String, double[]> genreFactors = new HashMap<>();
    private static Set<String> allGenres = new HashSet<>();
    private static Set<String> allUsers = new HashSet<>();

    public static void main(String[] args) throws Exception {
        List<Rating> ratings = loadRatings("dataset/ratings_20MB.json");
        saveOriginalMatrixWithNulls(ratings, "dataset/avaliacoes_iniciais_com_nulls.json");
        initializeFactors(ratings);
        trainModel(ratings);
        Map<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings);
        printRatingsMatrix(matrix, ratings);
        savePredictedRatingsToJson(ratings, matrix, "dataset/predicted_ratings.json");
    }

    private static List<Rating> loadRatings(String filename) throws IOException {
        Gson gson = new Gson();
        return gson.fromJson(new FileReader(filename), new TypeToken<List<Rating>>(){}.getType());
    }

    private static void initializeFactors(List<Rating> ratings) {
        for (Rating r : ratings) {
            allUsers.add(r.getUserId());
            allGenres.addAll(r.getGenres());
        }
        Random rand = new Random();
        for (String user : allUsers) {
            userFactors.put(user, rand.doubles(NUM_FEATURES, 0, 0.1).toArray());
        }
        for (String genre : allGenres) {
            genreFactors.put(genre, rand.doubles(NUM_FEATURES, 0, 0.1).toArray());
        }
    }

    private static void trainModel(List<Rating> ratings) {
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            for (Rating r : ratings) {
                String user = r.getUserId();
                for (String genre : r.getGenres()) {
                    double[] userVec = userFactors.get(user);
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

    private static Map<String, Map<String, Double>> predictRatingsMatrix(List<Rating> ratings) {
        Map<String, Map<String, Double>> matrix = new LinkedHashMap<>();

        for (String user : allUsers) {
            Map<String, Double> ratingsForUser = new LinkedHashMap<>();
            for (Rating r : ratings) {
                if (r.getUserId().equals(user)) {
                    ratingsForUser.put(r.getTitle(), r.getRating());
                }
            }
            for (Rating r : ratings) {
                if (!ratingsForUser.containsKey(r.getTitle())) {
                    double predicted = 0.0;
                    for (String genre : r.getGenres()) {
                        predicted += dot(userFactors.get(user), genreFactors.get(genre));
                    }
                    predicted /= r.getGenres().size();
                    ratingsForUser.put(r.getTitle(), predicted);
                }
            }
            matrix.put(user, ratingsForUser);
        }
        return matrix;
    }

    private static void printRatingsMatrix(Map<String, Map<String, Double>> matrix, List<Rating> ratings) {
        List<String> movies = ratings.stream().map(Rating::getTitle).distinct().collect(Collectors.toList());
        System.out.print("Usuário\\t");
        for (String movie : movies) {
            System.out.print(movie + "\t");
        }
        System.out.println();

        for (String user : allUsers) {
            System.out.print(user + "\t");
            Map<String, Double> userRatings = matrix.get(user);
            for (String movie : movies) {
                System.out.printf("%.2f\t", userRatings.getOrDefault(movie, 0.0));
            }
            System.out.println();
        }
    }

    private static void savePredictedRatingsToJson(
            List<Rating> ratings,
            Map<String, Map<String, Double>> predictedMatrix,
            String filename
    ) {
        List<String> allUsersOrdered = ratings.stream()
                .map(Rating::getUserId)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        List<String> allTitlesOrdered = ratings.stream()
                .map(Rating::getTitle)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        Map<String, List<String>> genreMap = new HashMap<>();
        for (Rating r : ratings) {
            genreMap.putIfAbsent(r.getTitle(), r.getGenres());
        }

        List<Map<String, Object>> output = new ArrayList<>();

        for (String user : allUsersOrdered) {
            Map<String, Object> userEntry = new LinkedHashMap<>();
            userEntry.put("user_id", user);

            List<Map<String, Object>> movieList = new ArrayList<>();
            Map<String, Double> userRatings = predictedMatrix.getOrDefault(user, Map.of());

            for (String title : allTitlesOrdered) {
                Map<String, Object> movieData = new LinkedHashMap<>();
                movieData.put("title", title);
                movieData.put("genre", genreMap.getOrDefault(title, List.of()));
                Double rating = userRatings.get(title);
                movieData.put("rating", rating != null ? rating : "null");
                movieList.add(movieData);
            }

            userEntry.put("movies", movieList);
            output.add(userEntry);
        }

        try (FileWriter writer = new FileWriter(filename)) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(output, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void saveOriginalMatrixWithNulls(List<Rating> ratings, String filename) {
        // Ordem garantida
        List<String> allUsersOrdered = ratings.stream()
                .map(Rating::getUserId)
                .distinct()
                .sorted()
                .toList();

        List<String> allTitlesOrdered = ratings.stream()
                .map(Rating::getTitle)
                .distinct()
                .sorted()
                .toList();

        Map<String, List<String>> genreMap = new HashMap<>();
        for (Rating r : ratings) {
            genreMap.putIfAbsent(r.getTitle(), r.getGenres());
        }

        // Cria um mapa de (usuário -> mapa de filmes avaliados)
        Map<String, Map<String, Double>> realRatingsMap = new HashMap<>();
        for (Rating r : ratings) {
            realRatingsMap
                    .computeIfAbsent(r.getUserId(), k -> new HashMap<>())
                    .put(r.getTitle(), r.getRating());
        }

        List<Map<String, Object>> output = new ArrayList<>();

        for (String user : allUsersOrdered) {
            Map<String, Object> userEntry = new LinkedHashMap<>();
            userEntry.put("user_id", user);

            List<Map<String, Object>> movieList = new ArrayList<>();
            for (String title : allTitlesOrdered) {
                Map<String, Object> movieData = new LinkedHashMap<>();
                movieData.put("title", title);
                movieData.put("genre", genreMap.getOrDefault(title, List.of()));
                Double rating = realRatingsMap.getOrDefault(user, Map.of()).get(title);
                movieData.put("rating", rating != null ? rating : "null");
                movieList.add(movieData);
            }
            userEntry.put("movies", movieList);
            output.add(userEntry);
        }

        try (FileWriter writer = new FileWriter(filename)) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(output, writer);
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