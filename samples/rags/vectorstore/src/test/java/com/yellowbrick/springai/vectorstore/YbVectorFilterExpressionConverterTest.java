package com.yellowbrick.springai.vectorstore;

import org.junit.jupiter.api.Test;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.ai.vectorstore.filter.Filter.Expression;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.AND;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.EQ;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.GTE;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.IN;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.LTE;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.NE;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.NIN;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.OR;

import org.springframework.ai.vectorstore.filter.Filter.Group;
import org.springframework.ai.vectorstore.filter.Filter.Key;
import org.springframework.ai.vectorstore.filter.Filter.Value;

import java.util.List;

class YbVectorFilterExpressionConverterTest {
    YbVectorFilterExpressionConverter converter = new YbVectorFilterExpressionConverter();

    @Test
    void doThing(){
        var b = new FilterExpressionBuilder();

        String vectorExpr = converter.convertExpression(b.eq("file_name","EmployeeHandbook.odf").build());
        assertThat(vectorExpr).isEqualTo("metadata:file_name::varchar = 'EmployeeHandbook.odf'");
        System.out.println(vectorExpr);
    }

    @Test
    public void tesEqAndGte() {
        // genre == "drama" AND year >= 2020
        String vectorExpr = this.converter
                .convertExpression(new Expression(AND, new Expression(EQ, new Key("genre"), new Value("drama")),
                        new Expression(GTE, new Key("year"), new Value(2020))));
        assertThat(vectorExpr).isEqualTo("metadata:genre::varchar = 'drama' AND metadata:year::varchar >= 2020");
    }

    @Test
    public void tesIn() {
        // genre in ["comedy", "documentary", "drama"]
        String vectorExpr = this.converter.convertExpression(
                new Expression(IN, new Key("genre"), new Value(List.of("comedy", "documentary", "drama"))));
        assertThat(vectorExpr)
                .isEqualTo("(metadata:genre::varchar = 'comedy' OR metadata:genre::varchar = 'documentary' OR metadata:genre::varchar = 'drama')");
    }

    @Test
    public void testNe() {
        // year >= 2020 OR country == "BG" AND city != "Sofia"
        String vectorExpr = this.converter
                .convertExpression(new Expression(OR, new Expression(GTE, new Key("year"), new Value(2020)),
                        new Expression(AND, new Expression(EQ, new Key("country"), new Value("BG")),
                                new Expression(NE, new Key("city"), new Value("Sofia")))));
        assertThat(vectorExpr).isEqualTo("metadata:year::varchar >= 2020 OR metadata:country::varchar = 'BG' AND metadata:city::varchar != 'Sofia'");
    }

    @Test
    public void testGroup() {
        // (year >= 2020 OR country == "BG") AND city NIN ["Sofia", "Plovdiv"]
        String vectorExpr = this.converter.convertExpression(new Expression(AND,
                new Group(new Expression(OR, new Expression(GTE, new Key("year"), new Value(2020)),
                        new Expression(EQ, new Key("country"), new Value("BG")))),
                new Expression(NIN, new Key("city"), new Value(List.of("Sofia", "Plovdiv")))));
        assertThat(vectorExpr)
                .isEqualTo("(metadata:year::varchar >= 2020 OR metadata:country::varchar = 'BG') AND !(metadata:city::varchar = 'Sofia' OR metadata:city::varchar = 'Plovdiv')");
    }
}