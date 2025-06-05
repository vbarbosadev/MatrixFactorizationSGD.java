package SGD;

import com.google.gson.*;
import com.google.gson.reflect.TypeToken;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.stream.Collectors;

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

        public String getUserId() { return user_id; }
        public String getTitle() { return title; }
        public List<String> getGenres() { return genre; }
        public double getRating() { return rating; }
    }

    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    private static ConcurrentHashMap<String, AtomicLongArray> userFactors = new ConcurrentHashMap<>();
    private static ConcurrentHashMap<String, AtomicLongArray> genreFactors = new ConcurrentHashMap<>();
    private static Set<String> allGenres = ConcurrentHashMap.newKeySet();
    private static Set<String> allUsers = ConcurrentHashMap.newKeySet();

    private static final Gson gson = new Gson();
    private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

    public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
        ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
        List<Future<?>> futures = new ArrayList<>();

        try (ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (String filename : filenames) {
                Future<?> future = virtualThreadExecutor.submit(() -> {
                    try (FileReader reader = new FileReader(filename)) {
                        List<Rating> localList = gson.fromJson(reader, ratingListType);
                        if (localList != null) {
                            ratings.addAll(localList);
                        }
                    } catch (IOException e) {
                        System.err.println("Erro ao ler arquivo: " + filename + " (" + e.getClass().getSimpleName() + ": " + e.getMessage() + ")");
                    } catch (JsonSyntaxException e) {
                        System.err.println("Erro de sintaxe JSON no arquivo: " + filename + " (" + e.getMessage() + ")");
                    }
                });
                futures.add(future);
            }
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (InterruptedException | ExecutionException e) {
                    System.err.println("Erro ao coletar o valor de Future: " + e);
                }
            }
        }
        return ratings;
    }

    public static void initializeFactors(ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        allUsers = ConcurrentHashMap.newKeySet();
        allGenres = ConcurrentHashMap.newKeySet();

        ExecutorService populator = Executors.newVirtualThreadPerTaskExecutor();
        for (Rating r : ratings) {
            populator.submit(() -> {
                allUsers.add(r.getUserId());
                allGenres.addAll(r.getGenres());
            });
        }
        populator.shutdown();
        populator.awaitTermination(1, TimeUnit.MINUTES);

        ExecutorService initializer = Executors.newVirtualThreadPerTaskExecutor();
        for (String user : allUsers) {
            initializer.submit(() -> {
                double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
                long[] longFeatures = new long[NUM_FEATURES];
                for (int i = 0; i < NUM_FEATURES; i++) {
                    longFeatures[i] = Double.doubleToRawLongBits(features[i]);
                }
                userFactors.put(user, new AtomicLongArray(longFeatures));
            });
        }
        for (String genre : allGenres) {
            initializer.submit(() -> {
                double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
                long[] longFeatures = new long[NUM_FEATURES];
                for (int i = 0; i < NUM_FEATURES; i++) {
                    longFeatures[i] = Double.doubleToRawLongBits(features[i]);
                }
                genreFactors.put(genre, new AtomicLongArray(longFeatures));
            });
        }
        initializer.shutdown();
        initializer.awaitTermination(1, TimeUnit.MINUTES);
    }

    public static void trainModel(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando Treinamento com atualizações atômicas.");
        List<Rating> ratingList = new ArrayList<>(ratings);
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            ratingList.parallelStream().forEach(rating -> {
                String user = rating.getUserId();
                AtomicLongArray userVec = userFactors.get(user);
                if (userVec == null) return;

                for (String genre : rating.getGenres()) {
                    AtomicLongArray genreVec = genreFactors.get(genre);
                    if (genreVec == null) continue;

                    double prediction = dot(userVec, genreVec);
                    double error = rating.getRating() - prediction;
                    updateVectors(userVec, genreVec, error);
                }
            });
        }
    }

    public static void updateVectors(AtomicLongArray userVec, AtomicLongArray genreVec, double error) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            long currentLongUser, newLongUser;
            do {
                currentLongUser = userVec.get(i);
                double u = Double.longBitsToDouble(currentLongUser);
                double g = Double.longBitsToDouble(genreVec.get(i));
                double newUserValue = u + LEARNING_RATE * (error * g - REGULARIZATION * u);
                newLongUser = Double.doubleToRawLongBits(newUserValue);
            } while (!userVec.compareAndSet(i, currentLongUser, newLongUser));

            long currentLongGenre, newLongGenre;
            do {
                currentLongGenre = genreVec.get(i);
                double g = Double.longBitsToDouble(currentLongGenre);
                double u = Double.longBitsToDouble(userVec.get(i));
                double newGenreValue = g + LEARNING_RATE * (error * u - REGULARIZATION * g);
                newLongGenre = Double.doubleToRawLongBits(newGenreValue);
            } while (!genreVec.compareAndSet(i, currentLongGenre, newLongGenre));
        }
    }

    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        ConcurrentMap<String, List<Rating>> ratingsByUser = ratings.stream().collect(Collectors.groupingByConcurrent(Rating::getUserId));
        ConcurrentMap<String, List<Rating>> ratingsByTitle = ratings.stream().collect(Collectors.groupingByConcurrent(Rating::getTitle));
        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (String user : allUsers) {
                executor.submit(() -> {
                    AtomicLongArray userVec = userFactors.get(user);
                    if (userVec == null) return;

                    Map<String, Double> userRatings = new HashMap<>();
                    List<Rating> knownRatings = ratingsByUser.get(user);
                    if (knownRatings != null) {
                        for (Rating r : knownRatings) {
                            userRatings.put(r.getTitle(), r.getRating());
                        }
                    }

                    for (Map.Entry<String, List<Rating>> entry : ratingsByTitle.entrySet()) {
                        String title = entry.getKey();
                        if (userRatings.containsKey(title)) continue;

                        List<Rating> ratingsForTitle = entry.getValue();
                        if (ratingsForTitle.isEmpty()) continue;

                        List<String> currentGenres = ratingsForTitle.get(0).getGenres();
                        if (currentGenres == null || currentGenres.isEmpty()) continue;

                        double predicted = 0.0;
                        int validGenres = 0;
                        for (String genre : currentGenres) {
                            AtomicLongArray genreVec = genreFactors.get(genre);
                            if (genreVec != null) {
                                predicted += dot(userVec, genreVec);
                                validGenres++;
                            }
                        }
                        if (validGenres > 0) {
                            userRatings.put(title, predicted / validGenres);
                        }
                    }
                    matrix.put(user, userRatings);
                });
            }
        }
        return matrix;
    }

    private static void printRatingsMatrix(ConcurrentMap<String, Map<String, Double>> matrix, ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        List<String> movies = ratings.stream().map(Rating::getTitle).distinct().sorted().collect(Collectors.toList());
        System.out.print("Usuário\t");
        movies.forEach(movie -> System.out.print(movie + "\t"));
        System.out.println();

        ConcurrentLinkedQueue<String> outputLines = new ConcurrentLinkedQueue<>();
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (String user : allUsers.stream().sorted().toList()) {
                executor.submit(() -> {
                    StringBuilder sb = new StringBuilder();
                    sb.append(user).append("\t");
                    Map<String, Double> userRatings = matrix.getOrDefault(user, Map.of());
                    for (String movie : movies) {
                        sb.append(String.format("%.2f\t", userRatings.getOrDefault(movie, 0.0)));
                    }
                    outputLines.add(sb.toString());
                });
            }
        }
        outputLines.forEach(System.out::println);
    }

    private static void savePredictedRatingsToMultipleFiles(ConcurrentLinkedQueue<Rating> ratings, Map<String, Map<String, Double>> predictedMatrix, String outputDirectory, String baseFilename) {
        List<String> allUsersOrdered = ratings.stream().map(Rating::getUserId).distinct().sorted().collect(Collectors.toList());
        List<String> allTitlesOrdered = ratings.stream().map(Rating::getTitle).distinct().sorted().collect(Collectors.toList());
        Map<String, List<String>> genreMap = ratings.stream().collect(Collectors.toConcurrentMap(Rating::getTitle, Rating::getGenres, (g1, g2) -> g1));
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        int usersPerFile = Math.max(1, allUsersOrdered.size() / 15);

        try (ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < allUsersOrdered.size(); i += usersPerFile) {
                int end = Math.min(i + usersPerFile, allUsersOrdered.size());
                List<String> userChunk = allUsersOrdered.subList(i, end);
                int filePart = (i / usersPerFile) + 1;
                String fullPath = String.format("%s/%s_part_%d.json", outputDirectory, baseFilename, filePart);

                Future<?> future = virtualThreadExecutor.submit(() -> {
                    List<Map<String, Object>> usersOutputForChunk = new ArrayList<>();
                    for (String user : userChunk) {
                        Map<String, Object> userEntry = new LinkedHashMap<>();
                        userEntry.put("user_id", user);
                        List<Map<String, Object>> movieList = new ArrayList<>();
                        Map<String, Double> userRatings = predictedMatrix.getOrDefault(user, Map.of());
                        for (String title : allTitlesOrdered) {
                            Map<String, Object> movieData = new LinkedHashMap<>();
                            movieData.put("title", title);
                            movieData.put("genre", genreMap.getOrDefault(title, List.of()));
                            movieData.put("rating", userRatings.get(title));
                            movieList.add(movieData);
                        }
                        userEntry.put("movies", movieList);
                        usersOutputForChunk.add(userEntry);
                    }
                    try (FileWriter writer = new FileWriter(fullPath)) {
                        gson.toJson(usersOutputForChunk, writer);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
                futures.add(future);
            }
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static double dot(AtomicLongArray a, AtomicLongArray b) {
        double sum = 0.0;
        final int length = a.length();
        for (int i = 0; i < length; i++) {
            sum += Double.longBitsToDouble(a.get(i)) * Double.longBitsToDouble(b.get(i));
        }
        return sum;
    }

    @Deprecated
    public static double dot(double[] a, double[] b) {
        throw new UnsupportedOperationException("Use a versão atômica de dot");
    }

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        Set<String> arquivos = Set.of("dataset/ratings_50MB.json");

        ConcurrentLinkedQueue<Rating> ratings = loadRatingsParallel(arquivos);
        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);

        initializeFactors(ratings);
        long initialTime = System.nanoTime();
        System.out.printf("initializeFactors rodou em: %.2f segundos%n", (initialTime - readTime) / 1e9);

        trainModel(ratings);
        long trainingTime = System.nanoTime();
        System.out.printf("trainingModel rodou em: %.2f segundos%n", (trainingTime - initialTime) / 1e9);

        ConcurrentMap<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings);
        long matrixTime = System.nanoTime();
        System.out.printf("matrixGen rodou em: %.2f segundos%n", (matrixTime - trainingTime) / 1e9);

        // printRatingsMatrix(matrix, ratings);
        long printTime = System.nanoTime();

        String outputDir = "output_ratings";
        new java.io.File(outputDir).mkdirs();
        String baseFilename = "predicted_user_ratings";
        savePredictedRatingsToMultipleFiles(ratings, matrix, outputDir, baseFilename);
        long saveTime = System.nanoTime();
        System.out.printf("saveJson rodou em: %.2f segundos%n", (saveTime - printTime) / 1e9);

        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);
    }
}