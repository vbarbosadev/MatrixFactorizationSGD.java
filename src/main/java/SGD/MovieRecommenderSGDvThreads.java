package SGD;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;
import org.jetbrains.annotations.NotNull;


public class MovieRecommenderSGDvThreads {

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

    private static class RatingLoader {

        private static final Gson gson = new Gson();
        private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

        public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
            ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
            List<Thread> threads = new ArrayList<>();

            for (String filename : filenames) {
                Thread t = Thread.ofVirtual().start(() -> {
                    try (FileReader reader = new FileReader(filename)) {
                        List<Rating> localList = new Gson().fromJson(reader, new TypeToken<List<Rating>>() {}.getType());
                        ratings.addAll(localList);
                    } catch (IOException e) {
                        System.err.println("Erro ao ler arquivo: " + filename);
                        e.printStackTrace();
                    }
                });
                threads.add(t);
            }

            for (Thread t : threads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }

            return ratings;
        }


        // Sobrecarga para um único arquivo
        public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(String filename) {
            return loadRatingsParallel(Set.of(filename));
        }
    }

    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    private static ConcurrentHashMap<String, double[]> userFactors = new ConcurrentHashMap<>();
    private static ConcurrentHashMap<String, double[]> genreFactors = new ConcurrentHashMap<>();
    private static Set<String> allGenres = ConcurrentHashMap.newKeySet();
    private static Set<String> allUsers = ConcurrentHashMap.newKeySet();


    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        Set<String> arquivos = Set.of("dataset/ratings_20MB.json", "dataset/ratings_20MBv2.json");

        ConcurrentLinkedQueue<Rating> ratings = RatingLoader.loadRatingsParallel(arquivos);

        //saveOriginalMatrixWithNulls(ratings, "dataset/avaliacoes_iniciais_com_nulls.json");
        initializeFactors(ratings);
        trainModel(ratings);
        ConcurrentMap<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings);
        printRatingsMatrix(matrix, ratings);
        savePredictedRatingsToJson(ratings, matrix, "dataset/predicted_ratings.json");

        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);
    }


    private static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        allUsers = ConcurrentHashMap.newKeySet();
        allGenres = ConcurrentHashMap.newKeySet();

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {

            // Popular allUsers e allGenres em paralelo
            for (Rating r : ratings) {
                executor.submit(() -> {
                    allUsers.add(r.getUserId());
                    allGenres.addAll(r.getGenres());
                });
            }

            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        }

        // Inicializar userFactors e genreFactors em paralelo, usando ThreadLocalRandom
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {

            for (String user : allUsers) {
                executor.submit(() -> {
                    double[] features = ThreadLocalRandom.current()
                            .doubles(NUM_FEATURES, 0, 0.1)
                            .toArray();
                    userFactors.put(user, features);
                });
            }

            for (String genre : allGenres) {
                executor.submit(() -> {
                    double[] features = ThreadLocalRandom.current()
                            .doubles(NUM_FEATURES, 0, 0.1)
                            .toArray();
                    genreFactors.put(genre, features);
                });
            }

            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        }
    }



    private static void trainModel(ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        System.out.println("Iniciando Treinamento");
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
                for (Rating r : ratings) {
                    executor.submit(() -> {
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
                    });
                }
                executor.shutdown();
                executor.awaitTermination(5, TimeUnit.MINUTES);
            }
        }
    }


    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings) {
        ConcurrentMap<String, ConcurrentLinkedQueue<Rating>> ratingsByUser = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            ratingsByUser
                    .computeIfAbsent(r.getUserId(), k -> new ConcurrentLinkedQueue<>())
                    .add(r);
        }


        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();
        List<Thread> threads = new ArrayList<>();

        for (String user : allUsers) {
            Thread t = Thread.ofVirtual().start(() -> {
                Map<String, Double> ratingsForUser = new ConcurrentHashMap<>();
                ConcurrentLinkedQueue<Rating> userRatings = ratingsByUser.getOrDefault(user, new ConcurrentLinkedQueue<>());

                for (Rating r : userRatings) {
                    ratingsForUser.put(r.getTitle(), r.getRating());
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
            });
            threads.add(t);
        }

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        return matrix;
    }



    private static void printRatingsMatrix( ConcurrentMap<String, Map<String, Double>> matrix,
            ConcurrentLinkedQueue<Rating> ratings) {
        // Lista concorrente e thread-safe de filmes (ordem mantida pela lista normal)
        List<String> movies = ratings.stream()
                .map(Rating::getTitle)
                .distinct()
                .collect(Collectors.toList());

        System.out.print("Usuário\t");
        for (String movie : movies) {
            System.out.print(movie + "\t");
        }
        System.out.println();

        // Usar ConcurrentLinkedQueue para threads (virtual threads)
        List<Thread> threads = new ArrayList<>();
        ConcurrentLinkedQueue<String> outputLines = new ConcurrentLinkedQueue<>();

        for (String user : allUsers) {
            Thread t = Thread.ofVirtual().start(() -> {
                StringBuilder sb = new StringBuilder();
                sb.append(user).append("\t");

                Map<String, Double> userRatings = matrix.getOrDefault(user, Map.of());
                for (String movie : movies) {
                    sb.append(String.format("%.2f\t", userRatings.getOrDefault(movie, 0.0)));
                }

                outputLines.add(sb.toString());
            });
            threads.add(t);
        }

        // Espera terminar todas as threads
        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Imprime as linhas (ordem não garantida pois concorrente; se quiser ordem, pode ordenar)
        outputLines.forEach(System.out::println);
    }





    private static void savePredictedRatingsToJson(
            ConcurrentLinkedQueue<Rating> ratings,
            Map<String, Map<String, Double>> predictedMatrix,
            String filename
    ) {
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

        Map<String, List<String>> genreMap = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            genreMap.putIfAbsent(r.getTitle(), r.getGenres());
        }

        ConcurrentSkipListMap<String, Map<String, Object>> output = new ConcurrentSkipListMap<>();
        List<Thread> threads = new ArrayList<>();

        for (String user : allUsersOrdered) {
            Thread t = Thread.ofVirtual().start(() -> {
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
                output.put(user, userEntry);
            });
            threads.add(t);
        }

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        try (FileWriter writer = new FileWriter(filename)) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(output.values(), writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }




    private static double dot(double[] a, double[] b) {
        final int length = a.length;
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        int i = 0;
        for (; i <= length - 4; i += 4) {
            sum0 += a[i] * b[i];
            sum1 += a[i + 1] * b[i + 1];
            sum2 += a[i + 2] * b[i + 2];
            sum3 += a[i + 3] * b[i + 3];
        }
        double sum = sum0 + sum1 + sum2 + sum3;
        for (; i < length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

}