import Foundation

private func readFormFactor(_ url: URL, skipHeader: Bool = false) throws -> ([Float], [Float]) {
    let content = try String(contentsOf: url, encoding: .utf8)
    let lines = content.split(whereSeparator: \.isNewline).dropFirst(skipHeader ? 1 : 0)
    var q: [Float] = []
    var values: [Float] = []
    for line in lines {
        let columns = line.split(whereSeparator: \.isWhitespace)
        if columns.count >= 2, let x = Float(columns[0]), let y = Float(columns[1]) {
            q.append(x)
            values.append(y)
        }
    }
    guard q.count >= 2, zip(q, q.dropFirst()).allSatisfy({ $0.0 < $0.1 }) else {
        throw TransportError.message("invalid form-factor table: \(url.path)")
    }
    return (q, values)
}

private func interpolate(_ x: Float, xGrid: [Float], values: [Float]) -> Float {
    if x <= xGrid[0] { return values[0] }
    if x >= xGrid[xGrid.count - 1] { return values[values.count - 1] }
    var lower = 0
    var upper = xGrid.count - 1
    while lower + 1 < upper {
        let middle = (lower + upper) / 2
        if xGrid[middle] <= x { lower = middle } else { upper = middle }
    }
    let fraction = (x - xGrid[lower]) / (xGrid[upper] - xGrid[lower])
    return values[lower] * (1 - fraction) + values[upper] * fraction
}

private func normalizedCDF(
    energies: [Float], intensity: (_ q: Float) -> Float, label: String
) throws -> [Float] {
    var tables: [Float] = []
    tables.reserveCapacity(energies.count * 4097)
    for energy in energies {
        var cdf = [Float](repeating: 0, count: 4097)
        for index in 1...4096 {
            let theta = (Float(index) - 0.5) * .pi / 4096
            let q = 2 * energy / 510.99895 * sin(theta / 2)
            let cosine = cos(theta)
            cdf[index] = cdf[index - 1]
                + max(0, intensity(q) * (1 + cosine * cosine) * sin(theta))
        }
        guard cdf[4096] > 0 else {
            throw TransportError.message("empty \(label) Rayleigh distribution")
        }
        tables += cdf.map { $0 / cdf[4096] }
    }
    return tables
}

func sampleRayleighCDF(physics: Physics, formFactorURL: URL) throws -> [Float] {
    let (q, formFactor) = try readFormFactor(formFactorURL)
    return try normalizedCDF(energies: physics.energy_kev, intensity: { value in
        let factor = interpolate(value, xGrid: q, values: formFactor)
        return factor * factor
    }, label: "sample")
}

func airRayleighCDF(physics: Physics) throws -> [Float] {
    guard let dataRoot = ProcessInfo.processInfo.environment["G4LEDATA"] else {
        throw TransportError.message("G4LEDATA is required for air Rayleigh tables")
    }
    let directory = URL(fileURLWithPath: dataRoot)
        .appendingPathComponent("penelope/rayleigh")
    let (nitrogenQ, nitrogenFF) = try readFormFactor(
        directory.appendingPathComponent("pdaff07.p08"), skipHeader: true
    )
    let (oxygenQ, oxygenFF) = try readFormFactor(
        directory.appendingPathComponent("pdaff08.p08"), skipHeader: true
    )
    let oxygenToNitrogen = Float((0.3 / 15.9994) / (0.7 / 14.0067))
    return try normalizedCDF(energies: physics.energy_kev, intensity: { value in
        let nitrogen = interpolate(value, xGrid: nitrogenQ, values: nitrogenFF)
        let oxygen = interpolate(value, xGrid: oxygenQ, values: oxygenFF)
        return nitrogen * nitrogen + oxygenToNitrogen * oxygen * oxygen
    }, label: "air")
}
